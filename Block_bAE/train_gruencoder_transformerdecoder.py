import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from pytorch_lightning.strategies import DDPStrategy
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar
from swanlab.integration.pytorch_lightning import SwanLabLogger
from pathlib import Path
import swanlab
import logging
import selfies as sf
from rdkit import Chem, RDLogger
from tqdm import tqdm
from torch.nn.attention import SDPBackend, sdpa_kernel
# --- Import new Transformer model ---
from model_gru_transformer import BlockbAE_GRUEncoder_TransformerDecoder_Gumbel
from datamodule import BlockDataModule
RDLogger.DisableLog('rdApp.*')
log = logging.getLogger(__name__)

class SetEpochCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        if hasattr(trainer.datamodule, 'train_dataset') and hasattr(trainer.datamodule.train_dataset, 'set_epoch'):
             log.debug(f"Setting epoch for train_dataset to {trainer.current_epoch}")
             trainer.datamodule.train_dataset.set_epoch(trainer.current_epoch)

def log_reconstruction_samples(
    epoch: int,
    outputs: list[dict],
    id_to_token: dict[int, str],
    pad_id: int,
    sos_id: int,
    eos_id: int,
    num_samples_to_print: int = 5
):
    if not outputs:
        return

    log.info("\n" + "="*50)
    log.info(f"  EPOCH {epoch}: DECODING SAMPLES CHECK")
    log.info("="*50)

    first_batch_with_samples = next((o for o in outputs if isinstance(o, dict) and 'originals' in o), None)
    
    if not first_batch_with_samples:
        log.info("  No decoded samples found in this validation epoch's outputs.")
        log.info("="*50 + "\n")
        return

    originals_for_print = first_batch_with_samples['originals']
    recons_for_print = first_batch_with_samples['reconstructed']
    num_to_print = min(num_samples_to_print, originals_for_print.size(0))

    for i in range(num_to_print):
        orig_ids = [tid for tid in originals_for_print[i].tolist() if tid not in [pad_id, sos_id]]
        recon_ids = []
        for tid in recons_for_print[i].tolist():
            if tid in [pad_id, sos_id]: continue
            recon_ids.append(tid)
            if tid == eos_id: break
        
        log.info(f"\n--- Sample {i+1}/{num_to_print} ---")
        orig_str = " ".join([id_to_token.get(tid, f"UNK({tid})") for tid in orig_ids])
        recon_str = " ".join([id_to_token.get(tid, f"UNK({tid})") for tid in recon_ids])
        log.info(f"  ORIGINAL: {orig_str}")
        log.info(f"  RECONSTRUCTED: {recon_str}")
    log.info("="*50 + "\n")

class BitUsageCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.val_outputs = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if isinstance(outputs, dict) and 'q' in outputs:
            self.val_outputs.append(outputs['q'].detach().cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.val_outputs or not trainer.logger:
            self.val_outputs.clear()
            return
        all_q = torch.cat(self.val_outputs, dim=0)
        bit_usage = all_q.mean(0)
        pl_module.log_dict({
            "val_bit_usage_mean": bit_usage.mean().item(),
            "val_bit_usage_std": bit_usage.std().item(),
            "val_active_bits_per_sample": (all_q > 0.5).float().sum(1).mean().item()
        }, on_epoch=True, sync_dist=trainer.world_size > 1)
        self.val_outputs.clear()

class LitBlockbAE_Transformer(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        if not hasattr(pl, 'global_rank') or pl.utilities.rank_zero_only.rank == 0:
            print("\n" + "*"*20 + " [INIT] HYPERPARAMETER VERIFICATION " + "*"*20)
            # Directly check the original hparams dictionary
            ss_prob = hparams.get('scheduled_sampling_start_prob', '!!! KEY NOT FOUND !!!')
            ss_steps = hparams.get('ss_every_n_steps', '!!! KEY NOT FOUND !!!')
            print("*"*70 + "\n")

        model_args = {
        "vocab_size": self.hparams.vocab_size,
        "d_model": self.hparams.d_model,
        "nhead": self.hparams.nhead,
        "dim_feedforward": self.hparams.dim_feedforward,
        "num_encoder_layers": self.hparams.num_encoder_layers,
        "num_decoder_layers": self.hparams.num_decoder_layers,
        "dropout": self.hparams.dropout,
        "latent_dim": self.hparams.latent_dim,
        "initial_temp": self.hparams.gumbel_initial_temp,
        "pad_id": self.hparams.pad_id,
        "share_weights": self.hparams.get("share_embedding_weights", True),
        "encoder_type": self.hparams.get("encoder_type", "transformer")
    }
        self.model = BlockbAE_GRUEncoder_TransformerDecoder_Gumbel(**model_args)
        self._uncompiled_model = self.model
        try:
            self.model = torch.compile(self.model, mode="max-autotune")
            log.info("Model compiled successfully.")
        except Exception as e:
            log.warning(f"Model compilation failed: {e}. Using uncompiled model.")

        # [FINAL ROBUST FIX 2.0] No changes needed
        self.val_decode_batches = self.hparams.get('val_decode_batches', 16)
        self.train_decode_batches = self.hparams.get('train_decode_batches', 16)


        # Gumbel and Entropy parameter initialization, no changes needed
        total_epochs = self.hparams.get('max_epochs', 50)
        self.entropy_n1 = self.hparams.get('entropy_n1', int(total_epochs * 0.7))
        self.entropy_n2 = self.hparams.get('entropy_n2', int(total_epochs * 0.8))
        self.gumbel_tau_mid = self.hparams.get('gumbel_tau_mid', 0.5)
        self.gumbel_initial_temp = self.hparams.get('gumbel_initial_temp', 1.0)
        self.gumbel_temp_min = self.hparams.get('gumbel_temp_min', 0.4)
        # self.gumbel_temp_decay_rate = self.hparams.get('gumbel_temp_anneal_rate', 0.955)


        # ==============================
        self.scheduled_sampling_start_prob = self.hparams.get("scheduled_sampling_start_prob", 0.1)
        self.scheduled_sampling_end_prob = self.hparams.get("scheduled_sampling_end_prob", 0.5)
        self.scheduled_sampling_warmup_epochs = self.hparams.get("scheduled_sampling_warmup_epochs", 5)
        self.word_dropout_prob = self.hparams.get('word_dropout_prob', 0.15)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.hparams.pad_id, label_smoothing=0)

        
        self.id_to_token = None


    def setup(self, stage: str):
        if (stage == 'fit' or stage == 'validate') and self.id_to_token is None:
            datamodule = self.trainer.datamodule
            dataset = datamodule.train_dataset
            if isinstance(dataset, torch.utils.data.Subset):
                dataset = dataset.dataset
            self.id_to_token = {i: token for i, token in enumerate(dataset.vocab_data['itos_lst'])}

    @staticmethod
    def _bernoulli_entropy(q: torch.Tensor):
        q_float32 = q.float()
        eps = torch.finfo(q_float32.dtype).eps # Get the smallest positive number in float32
        q_clamped = torch.clamp(q_float32, min=eps, max=1.0 - eps)
        entropy = -(q_clamped * torch.log(q_clamped) + (1 - q_clamped) * torch.log(1 - q_clamped))
        return entropy.sum(dim=-1).to(q.dtype)

    def _calculate_losses(self, base_logits, recon_logits, trg_y):
        """
        A unified helper function for computing all loss terms.
        """
        clean_recon_logits = torch.nan_to_num(recon_logits, nan=0.0, posinf=1e4, neginf=-1e4)
        recon_loss = self.criterion(clean_recon_logits.reshape(-1, clean_recon_logits.size(-1)), trg_y.reshape(-1))
        
        kl_coeff = self._get_kl_coeff()

        prior_p = self.hparams.get("kl_prior_p", 0.5)
        q_prob = torch.sigmoid(base_logits.float())

        eps = 1e-8
        q_prob = torch.clamp(q_prob, eps, 1.0 - eps)

        p_tensor = torch.full_like(q_prob, prior_p)
        kl_per_bit = q_prob * (torch.log(q_prob) - torch.log(p_tensor)) + \
                    (1 - q_prob) * (torch.log(1 - q_prob) - torch.log(1 - p_tensor))

        # Sum over bit dimension to get KL loss per sample
        kl_per_sample = kl_per_bit.sum(dim=-1)

        raw_kl_mean = kl_per_sample.mean()

        # Apply free-bits
        free_bits = self.hparams.get("kl_free_bits", 32) # No effect if set to 0
        kl_after_fb = F.relu(kl_per_sample - free_bits)

        # Average over batch dimension to get final KL loss base value
        kl_loss_base = kl_after_fb.mean()

        # Multiply by annealing coefficient
        kl_loss = kl_coeff * kl_loss_base

        # 3. Compute total loss
        total_loss = recon_loss + kl_loss

        # 4. Return dictionary
        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "raw_kl": raw_kl_mean,
            "kl_loss_base": kl_loss_base,
            "kl_coeff": kl_coeff   # Record current coefficient for monitoring
        }


    def _scheduled_sampling_prob(self):
        start_prob = self.scheduled_sampling_start_prob
        end_prob = self.scheduled_sampling_end_prob
        warmup_epochs = self.scheduled_sampling_warmup_epochs

        if self.current_epoch >= warmup_epochs:
            return end_prob
        else:
            progress = self.current_epoch / warmup_epochs
            return start_prob + (end_prob - start_prob) * progress

    def _scheduled_sampling_step(self, batch: dict[str, torch.Tensor]):
        """
        Expensive training step specifically for Scheduled Sampling.
        """
        src = batch['src']
        trg, trg_y = src[:, :-1], src[:, 1:]
        trg_for_decode =  trg.clone()  # Create copy to avoid in-place modification
        # Word Dropout can still be applied to make the task more challenging
        if self.training and self.word_dropout_prob > 0:
            trg_for_decode = self._apply_bert_style_word_dropout(trg_for_decode)
        else:
            trg_for_decode = trg_for_decode

        base_logits, z = self.model.encode(src=src)

        # --- Theoretically correct SS implementation ---
        # First decode (no gradient) to get model's own predictions
        with torch.no_grad():
            tmp_logits = self._uncompiled_model.decode(z=z, trg=trg_for_decode)
            predicted_ids = tmp_logits.argmax(-1)

        # Replace part of the true tokens with predictions
        final_trg = trg_for_decode.clone() # Create copy to avoid in-place modification
        prob = torch.rand_like(final_trg.float())
        ss_prob = self._scheduled_sampling_prob()
        sampling_mask = (prob < ss_prob) & (final_trg != self.hparams.pad_id)
        final_trg[sampling_mask] = predicted_ids[sampling_mask]

        # Second decode (with gradient) using the corrupted final_trg to compute final logits
        recon_logits = self.model.decode(z=z, trg=final_trg)

        # --- Loss computation ---
        loss_dict = self._calculate_losses(base_logits, recon_logits, trg_y) 
        with torch.no_grad():
            q_monitor = torch.sigmoid(base_logits)
        loss_dict.update({'q': q_monitor, 'z': z})
        
        return loss_dict

    def _compute_and_log_metrics(self, gathered_originals, gathered_reconstructed, prefix: str):
        """
        Unified helper function for computing and logging reconstruction metrics.
        prefix: 'val' or 'train'
        """
        # 1. Print samples
        if gathered_originals: # Ensure data exists
            rank_0_outputs = [{'originals': gathered_originals[0][i], 'reconstructed': gathered_reconstructed[0][i]}
                            for i in range(len(gathered_originals[0]))]
            log_reconstruction_samples(
                epoch=self.current_epoch, outputs=rank_0_outputs, id_to_token=self.id_to_token,
                pad_id=self.hparams.pad_id, sos_id=self.hparams.sos_id, eos_id=self.hparams.eos_id
            )

        # 2. Compute metrics
        total_seqs, perfect_recons, total_tokens, correct_tokens = 0, 0, 0, 0

        special_tokens = {self.hparams.sos_id, self.hparams.pad_id}

        for originals_per_proc, recons_per_proc in zip(gathered_originals, gathered_reconstructed):
            for originals_batch, recons_batch in zip(originals_per_proc, recons_per_proc):
                for i in range(originals_batch.size(0)):
                    total_seqs += 1
                    # Extract original sequence (remove special tokens)
                    orig_ids = [tid.item() for tid in originals_batch[i] if tid.item() not in special_tokens]

                    # Extract reconstructed sequence (remove special tokens, stop at EOS)
                    recon_ids = []
                    found_eos = False
                    for tid in recons_batch[i]:
                        tid_item = tid.item()
                        if tid_item in special_tokens: continue
                        if tid_item == self.hparams.eos_id:
                            found_eos = True
                            break
                        recon_ids.append(tid_item)

                    if found_eos:
                        recon_ids.append(self.hparams.eos_id)

                    # Perfect match check
                    if orig_ids == recon_ids:
                        perfect_recons += 1

                    # Token-level accuracy
                    total_tokens += len(orig_ids)
                    for k in range(min(len(orig_ids), len(recon_ids))):
                        if orig_ids[k] == recon_ids[k]:
                            correct_tokens += 1

        # 3. Log metrics
        perfect_match_rate = (perfect_recons / total_seqs) * 100.0 if total_seqs > 0 else 0.0
        token_accuracy = (correct_tokens / total_tokens) * 100.0 if total_tokens > 0 else 0.0
        

        is_val = (prefix == 'val')
        self.log(f'{prefix}_reconstruction_rate', perfect_match_rate, on_epoch=True, prog_bar=True, rank_zero_only=True)
        self.log(f'{prefix}_token_accuracy', token_accuracy, on_epoch=True, prog_bar=True, rank_zero_only=True)
        
        log.info(f"{prefix.capitalize()} perfect match rate (global): {perfect_match_rate:.2f}% ({perfect_recons}/{total_seqs})")
        log.info(f"{prefix.capitalize()} token accuracy (global): {token_accuracy:.2f}% ({correct_tokens}/{total_tokens})")

        for handler in logging.getLogger().handlers:
            handler.flush()

    def _apply_bert_style_word_dropout(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Apply BERT-style Word Dropout to input token sequences.
        Only operates on non-special tokens.
        """
        # Copy input to avoid in-place modification
        masked_tokens = tokens.clone()

        # 1. Determine candidate positions that can be masked
        # (not PAD, SOS, EOS)
        candidate_mask = (tokens != self.hparams.pad_id) & \
                         (tokens != self.hparams.sos_id) & \
                         (tokens != self.hparams.eos_id)

        # 2. Calculate number of tokens to mask (e.g., 15% of candidate positions)
        # and randomly select these positions
        num_to_mask = int(torch.sum(candidate_mask) * self.word_dropout_prob)
        if num_to_mask == 0:
            return masked_tokens # Return directly if no tokens can be masked

        candidate_indices = torch.where(candidate_mask)
        # Randomly select num_to_mask from candidate indices
        rand_indices = torch.randperm(candidate_indices[0].size(0), device=tokens.device)[:num_to_mask]

        # Get final 2D indices for mask operation
        mask_indices_b = candidate_indices[0][rand_indices]
        mask_indices_t = candidate_indices[1][rand_indices]

        # 3. Apply 70-20-10 rule to selected positions
        # Generate random number from 0 to 1 to decide replacement strategy
        rand_decision = torch.rand(num_to_mask, device=tokens.device)

        # 70% case -> replace with UNK
        mask_80 = rand_decision < 0.7
        masked_tokens[mask_indices_b[mask_80], mask_indices_t[mask_80]] = self.hparams.unk_id

        # 10% case -> replace with random word from vocabulary
        mask_10_rand = (rand_decision >= 0.7) & (rand_decision < 0.9)
        num_rand_replace = torch.sum(mask_10_rand)
        if num_rand_replace > 0:
            random_words = torch.randint(
                low=4, # Assume 0,1,2,3 are special tokens (SOS, EOS, UNK, PAD), start from 4
                high=self.hparams.vocab_size,
                size=(num_rand_replace,),
                device=tokens.device
            )
            masked_tokens[mask_indices_b[mask_10_rand], mask_indices_t[mask_10_rand]] = random_words

        # 10% case -> keep original (no operation needed)
        # mask_10_keep = rand_decision >= 0.9

        return masked_tokens

    def _shared_step(self, batch: dict[str, torch.Tensor]):
        src = batch['src']
        trg, trg_y = src[:, :-1], src[:, 1:]
        trg_for_decode = trg.clone()

        # Word Dropout as the main input noise regularization
        if self.training and self.word_dropout_prob > 0:
            trg_for_decode = self._apply_bert_style_word_dropout(trg_for_decode)
        else:
            trg_for_decode = trg_for_decode

        # --- Core modification: perform only one complete forward pass ---
        base_logits, z = self.model.encode(src=src)
        recon_logits = self.model.decode(z=z, trg=trg_for_decode)
        
        loss_dict = self._calculate_losses(base_logits, recon_logits, trg_y)
        with torch.no_grad():
            q_monitor = torch.sigmoid(base_logits) 
        loss_dict.update({'q': q_monitor, 'z': z})
        return loss_dict

    def training_step(self, batch, batch_idx):
        use_ss = (
            self.scheduled_sampling_start_prob > 0 and
            # Enable from second epoch onwards to give model some stability time
            self.current_epoch > 50 and
            # Use global_step to ensure synchronization in DDP environment
            self.global_step % self.hparams.get("ss_every_n_steps", 2) == 0
        )

        if use_ss:
            # Call expensive SS path
            outputs = self._scheduled_sampling_step(batch)
        else:
            # Call regular, efficient path
            outputs = self._shared_step(batch)
        
        self.log_dict({
            "train/loss": outputs["loss"],
            "train/recon_loss": outputs["recon_loss"],
            "train/kl_loss": outputs["kl_loss"],
            "debug/raw_kl": outputs["raw_kl"],
            "debug/kl_loss_base": outputs["kl_loss_base"],
            "debug/kl_coeff": outputs["kl_coeff"]
        }, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return outputs["loss"]

    def validation_step(self, batch, batch_idx):
        outputs = self._shared_step(batch)
        self.log_dict({
            'val/loss': outputs["loss"],
            'val/recon_loss': outputs["recon_loss"]
        }, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return {'q': outputs['q'],'val/loss': outputs["loss"]}

    def _gumbel_temperature_update(self):
        """
        Implement a two-stage Gumbel temperature annealing strategy:
        1. From epoch 0 to n1, temperature exponentially decays from initial_temp to temp_min.
        2. After epoch n1, temperature remains at temp_min.
        """
        # Get current epoch and related hyperparameters
        current_epoch = self.current_epoch
        initial_temp = self.hparams.gumbel_initial_temp
        min_temp = self.hparams.gumbel_temp_min

        # n1 is the epoch where annealing ends
        anneal_end_epoch = self.hparams.entropy_n1

        # Determine which stage we're in
        if current_epoch < anneal_end_epoch:
            # --- Stage 1: Exponential decay period ---

            # To make temperature reach min_temp exactly at anneal_end_epoch,
            # we need to calculate the correct decay rate.
            # Formula: initial_temp * (rate ^ anneal_end_epoch) = min_temp
            # -> rate = (min_temp / initial_temp) ^ (1 / anneal_end_epoch)

            # Avoid division by zero or taking roots of negative numbers
            if initial_temp <= 0 or min_temp <= 0 or anneal_end_epoch == 0:
                return min_temp

            rate = (min_temp / initial_temp) ** (1.0 / anneal_end_epoch)

            # Calculate temperature for current epoch
            new_temp = initial_temp * (rate ** current_epoch)

            # Ensure temperature doesn't accidentally fall below minimum
            return max(new_temp, min_temp)
        else:
            # --- Stage 2: Hold period ---
            return min_temp
    
    def _get_kl_coeff(self):
        """
        Implement KL divergence weight annealing strategy (KL Cost Annealing).
        This is key to preventing posterior collapse.
        """
        # Get KL hyperparameters from hparams
        kl_lambda = self.hparams.get("kl_lambda", 0.00025)
        warmup_epochs = self.hparams.get("kl_warmup_epochs", 15)

        # During warmup period, KL weight grows linearly from 0 to target value kl_lambda
        if self.current_epoch < warmup_epochs:
            # Calculate warmup progress
            progress = self.current_epoch / max(1, warmup_epochs - 1)
            kl_coeff = kl_lambda * min(max(progress, 0.0), 1.0)
        else:
            # After warmup period, maintain fixed lambda value
            kl_coeff = kl_lambda

        return kl_coeff

    def on_train_epoch_start(self):
        # Moving Gumbel temperature update to on_train_epoch_start is better
        # because this ensures temperature is updated before the first training step of the epoch
        new_temp = self._gumbel_temperature_update()
        self.model.temperature.data.fill_(new_temp)
        self.log("debug/gumbel_temp", new_temp, on_step=False, on_epoch=True)

        # Can also log KL coefficient here if desired
        kl_coeff = self._get_kl_coeff()
        self.log("debug/kl_coeff_epoch_start", kl_coeff, on_step=False, on_epoch=True)

    # [COMPATIBILITY FIX] Change signature back and use self.validation_step_outputs
    def on_validation_epoch_end(self):
        current_epoch = self.current_epoch + 1
        if current_epoch % 3 != 0:
            log.info(f"Epoch {self.current_epoch}: Skipping reconstruction evaluation (only runs every 3 epochs).")
            return

        # [ALL_GATHER FIX] Only proceed if we have something to gather
        log.info(f"Epoch {self.current_epoch}: Starting full validation set reconstruction evaluation...")
        try:
            val_loader = self.trainer.datamodule.val_dataloader()
        except Exception as e:
            log.error(f"Could not get validation dataloader in on_validation_epoch_end: {e}")
            return
        all_originals_local = []
        all_reconstructed_local = []

        # 3. Iterate through entire validation set for decoding
        with torch.no_grad():
            for batch in val_loader:
                # Move data to current device
                src = batch['src'].to(self.device)

                # Use same encoding logic as during training
                base_logits, z = self.model.encode(src=src)

                # Use greedy_decode for reconstruction
                reconstructed_ids = self._uncompiled_model.greedy_decode(
                    z=z,
                    sos_id=self.hparams.sos_id,
                    eos_id=self.hparams.eos_id,
                    max_len=self.hparams.max_len
                )

                # Collect current process results (keep on GPU for all_gather)
                all_originals_local.append(src)
                all_reconstructed_local.append(reconstructed_ids)

        # 4. In DDP environment, gather results from all processes
        if self.trainer.world_size > 1:
            # all_gather aggregates list of lists
            gathered_originals_gpu = self.all_gather(all_originals_local)
            gathered_reconstructed_gpu = self.all_gather(all_reconstructed_local)
        else:
            # Single GPU environment, manually wrap in list to maintain consistent data structure
            gathered_originals_gpu = [all_originals_local]
            gathered_reconstructed_gpu = [all_reconstructed_local]

        # 5. Only perform computation and logging on rank 0 process
        if self.trainer.is_global_zero:
            log.info("Rank 0: Moving all gathered tensors to CPU for metrics calculation...")

            # Move aggregated data from GPU to CPU to free memory and perform computation
            # gathered_..._gpu structure: [ [proc0_batch0, proc0_batch1,...], [proc1_batch0, ...], ... ]
            cpu_originals = [[batch.cpu() for batch in proc_data] for proc_data in gathered_originals_gpu]
            cpu_reconstructed = [[batch.cpu() for batch in proc_data] for proc_data in gathered_reconstructed_gpu]

            # Free large tensors on GPU
            del gathered_originals_gpu, gathered_reconstructed_gpu, all_originals_local, all_reconstructed_local

            # Call metrics computation function to calculate and log reconstruction rate for entire validation set
            self._compute_and_log_metrics(cpu_originals, cpu_reconstructed, "val")

        # 2. Training set evaluation logic (moved from on_train_epoch_end)
        if self.hparams.train_decode_batches > 0:
            log.info("Starting end-of-epoch evaluation on training data...")
            try:
                # --- Get data ---
                train_decode_originals_local = []
                train_decode_recons_local = []
                train_eval_loader = self.trainer.datamodule.train_eval_dataloader()
                train_loader_iter = iter(train_eval_loader)

                with torch.no_grad():
                    for _ in range(self.hparams.train_decode_batches):
                        try:
                            batch = next(train_loader_iter)
                        except StopIteration:
                            log.warning("Train dataloader exhausted before reaching train_decode_batches.")
                            break

                        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                        src = batch['src']
                        base_logits_val, z_val = self.model.encode(src=src)
                        reconstructed_ids = self._uncompiled_model.greedy_decode(
                            z=z_val, sos_id=self.hparams.sos_id, eos_id=self.hparams.eos_id, max_len=self.hparams.max_len
                        )
                        train_decode_originals_local.append(src)
                        train_decode_recons_local.append(reconstructed_ids)

                # --- Normal flow: gather, transfer to CPU, compute metrics ---
                if not train_decode_originals_local:
                    if self.trainer.is_global_zero:
                        log.warning("No training samples were decoded for evaluation.")
                    return
                gathered_train_originals = self.all_gather(train_decode_originals_local)
                gathered_train_reconstructed = self.all_gather(train_decode_recons_local)

                if self.trainer.is_global_zero:
                    log.info("Moving gathered training tensors to CPU for metrics calculation.")
                    cpu_originals = [[batch.cpu() for batch in proc_data] for proc_data in gathered_train_originals]
                    cpu_reconstructed = [[batch.cpu() for batch in proc_data] for proc_data in gathered_train_reconstructed]
                    del gathered_train_originals
                    del gathered_train_reconstructed

                    self._compute_and_log_metrics(cpu_originals, cpu_reconstructed, "train")

            except Exception as e:
                log.error(f"Failed during training set evaluation in on_validation_epoch_end: {e}", exc_info=True)

    def configure_optimizers(self):

        base_lr = float(self.hparams.learning_rate)

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=base_lr,                     # Use target lr here, warmup phase uses LinearLR multiplier
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-6
        )

        try:
            total_steps = int(self.trainer.estimated_stepping_batches)
        except Exception:
            train_loader = self.trainer.datamodule.train_dataloader()
            steps_per_epoch = len(train_loader)
            accum = getattr(self.trainer, "accumulate_grad_batches", 1)
            total_steps = max(1, steps_per_epoch * self.trainer.max_epochs // accum)

        # --- Warmup/main training steps ---
        warmup_steps = int(total_steps * 0.10)
        warmup_steps = max(0, warmup_steps)       # Safety check
        main_steps = max(1, total_steps - warmup_steps)

        schedulers = []
        milestones = []

        # 1) Warmup (optional: skip if warmup_steps==0)
        if warmup_steps > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=1e-3,   # Initial lr = base_lr * 1e-3
                end_factor=1.0,
                total_iters=warmup_steps
            )
            schedulers.append(warmup_scheduler)
            milestones.append(warmup_steps)

        # 2) Cosine decay (to a small fraction of base_lr, not 0)
        final_ratio = 0.05                       # End at ~5% of base_lr
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=main_steps,
            eta_min=base_lr * final_ratio
        )
        schedulers.append(cosine_scheduler)

        # If only one scheduler (e.g., warmup_steps==0), don't use SequentialLR
        if len(schedulers) == 1:
            scheduler = schedulers[0]
        else:
            scheduler = SequentialLR(
                optimizer,
                schedulers=schedulers,
                milestones=milestones  # Only one milestone=warmup_steps
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",   # Update per step
                "frequency": 1,
            },
        }
    
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('high')

    parser = argparse.ArgumentParser()
    # --- Training settings ---
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)

    # --- Path settings ---
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    # --- Model architecture ---
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--dim_feedforward", type=int, default=1024, help="Dimension of FFN")
    parser.add_argument("--num_encoder_layers", type=int, default=3)
    parser.add_argument("--num_decoder_layers", type=int, default=3)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--max_len", type=int, default=96)
    parser.add_argument("--encoder_type", type=str, default="gru",
                        choices=['transformer', 'gru'],
                        help="Type of encoder to use.")

    # --- Optimizer and scheduler ---
    parser.add_argument("--lr", type=float, default=0.0005, dest="learning_rate")
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # --- Regularization ---
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--word_dropout_prob", type=float, default=0.2, help="Word Dropout probability")
    parser.add_argument("--kl_lambda", type=float, default=0.00005, help="Final weight of the KL divergence loss.")
    parser.add_argument("--kl_warmup_epochs", type=int, default=20, help="Epochs to ramp up KL loss from 0 to kl_lambda.")
    parser.add_argument("--kl_prior_p", type=float, default=0.5, help="The prior probability for the Bernoulli distribution (0.5 for uniform).")
    parser.add_argument("--kl_free_bits", type=float, default=32, help="Free bits per sample. KL loss is only applied if it exceeds this value (in nats).")

    # --- Gumbel annealing ---
    parser.add_argument("--gumbel_initial_temp", type=float, default=1.0)
    parser.add_argument("--gumbel_temp_min", type=float, default=0.4)
    parser.add_argument("--gumbel_tau_mid", type=float, default=0.5)
    parser.add_argument("--entropy_n1", type=int, default=None)
    parser.add_argument("--entropy_n2", type=int, default=None)

    # --- Scheduled Sampling ---
    parser.add_argument("--ss_every_n_steps", type=int, default=5, help="Run SS every N global steps")
    parser.add_argument("--scheduled_sampling_start_prob", type=float, default=0.1) # Recommend starting from 0
    parser.add_argument("--scheduled_sampling_end_prob", type=float, default=0.5) # Recommend not too high
    parser.add_argument("--scheduled_sampling_warmup_epochs", type=int, default=20, help="Warmup epochs for SS probability")

    # --- Other ---
    parser.add_argument("--val_decode_batches", type=int, default=15)
    parser.add_argument("--train_decode_batches", type=int, default=15)
    parser.add_argument("--val_split", type=float, default=0.01)
    parser.add_argument("--swanlab_project", type=str, default="BlockbAE-Project-Selfies")
    parser.add_argument("--swanlab_experiment_name", type=str, default="selfies")
    parser.add_argument("--no_share_embedding_weights", dest="share_embedding_weights", action="store_false", help="Disable weight sharing between embedding and output layer")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to a checkpoint to resume training from.")
    args = parser.parse_args()

    if args.entropy_n1 is None: args.entropy_n1 = int(args.max_epochs * 0.9)
    if args.entropy_n2 is None: args.entropy_n2 = int(args.max_epochs * 1)

    dm = BlockDataModule(data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, max_len=args.max_len)
    dm.prepare_data()
    dm.setup()

    hparams = vars(args)
    hparams['vocab_size'] = dm.vocab_size
    hparams['pad_id'] = dm.pad_id
    hparams['sos_id'] = dm.train_dataset.sos_id
    hparams['eos_id'] = dm.train_dataset.eos_id
    hparams['unk_id'] = dm.train_dataset.unk_id

    model = LitBlockbAE_Transformer(**hparams)

    swanlab_logger = SwanLabLogger(project=args.swanlab_project, experiment_name=args.swanlab_experiment_name, save_dir=args.save_dir)
    checkpoint_dir = Path(swanlab_logger.save_dir) / "checkpoints"
    
    callbacks = [
        ModelCheckpoint(dirpath=checkpoint_dir, monitor='val/recon_loss', mode='min', save_top_k=-1, filename='best-model-{epoch:02d}-{val/recon_loss:.4f}'),
        EarlyStopping(monitor='val/loss', patience=70, mode='min',check_on_train_epoch_end=False),
        LearningRateMonitor(logging_interval='step'),
        BitUsageCallback(),
        SetEpochCallback(),
        RichProgressBar()
    ]


    if args.devices > 1:
        strategy = DDPStrategy(find_unused_parameters=False, static_graph=False, gradient_as_bucket_view=True)
    else:
        strategy = "auto"
    trainer = pl.Trainer(
        accelerator=args.accelerator, devices=args.devices, max_epochs=args.max_epochs,
        callbacks=callbacks, logger=swanlab_logger, 
        gradient_clip_val=1.0,
        precision="bf16-mixed", strategy=strategy,
        # limit_train_batches=0.01,
        # limit_val_batches=0.1,
        num_sanity_val_steps=0
    )

    if args.resume_from_checkpoint:
        print("-" * 50)
        log.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        log.info("Loading hyperparameters from the checkpoint to override defaults.")

        # 1. Load hparams from checkpoint
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=torch.device('cpu'))
        hparams = checkpoint['hyper_parameters']


        original_ss_warmup = hparams.get('scheduled_sampling_warmup_epochs', 'N/A')
        new_ss_warmup = 30 # <--- This is the new value you want
        original_ss_start = hparams.get('scheduled_sampling_start_prob', 'N/A')
        new_ss_start = 0
        original_ss_end = hparams.get('scheduled_sampling_end_prob', 'N/A')
        new_ss_end = 0
        hparams['scheduled_sampling_end_prob'] = new_ss_end
        original_word_dropout = hparams.get('word_dropout_prob', 'N/A')
        new_word_dropout = 0.15

        original_max_epochs = hparams.get('max_epochs', 'N/A')
        new_max_epochs = 39
        hparams['max_epochs'] = new_max_epochs
        log.info(f"Overriding 'max_epochs': From {original_max_epochs} -> To {new_max_epochs}")

        hparams['scheduled_sampling_warmup_epochs'] = new_ss_warmup
        hparams['word_dropout_prob'] = new_word_dropout
        hparams['scheduled_sampling_start_prob'] = new_ss_start
        log.info(f"Overriding 'scheduled_sampling_warmup_epochs': From {original_ss_warmup} -> To {new_ss_warmup}")
        log.info(f"Overriding 'word_dropout_prob': From {original_word_dropout} -> To {new_word_dropout}")
        log.info(f"Overriding 'scheduled_sampling_start_prob': From {original_ss_start} -> To {new_ss_start}")
        log.info(f"Overriding 'scheduled_sampling_end_prob': From {original_ss_end} -> To {new_ss_end}")

        hparams.update(vars(args))

        print("-" * 50)
    else:
        # If not resuming from checkpoint, use the original flow
        hparams = vars(args)

    args.max_epochs = new_max_epochs
    log.info(f"Starting training for up to {args.max_epochs} epochs with Transformer model...")
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        trainer.fit_loop.max_epochs = 39
        trainer.fit(model, datamodule=dm, ckpt_path=args.resume_from_checkpoint)
    log.info("Training finished.")
    

if __name__ == '__main__':
    main()

