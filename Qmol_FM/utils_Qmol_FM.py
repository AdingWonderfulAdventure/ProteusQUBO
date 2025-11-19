# File: utils_Qmol_FM.py
import logging
import math
import torch, h5py, numpy as np, json, sys
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed, cpu_count, effective_n_jobs
from typing import List, Dict, Optional
import torch.multiprocessing as mp
import warnings
import torch.nn.functional as F

# --- VAE Model Import ---
try:
    Block_bAE_PARENT_DIR = "/root/workspace/ProteusQUBO/Block_bAE"
    sys.path.insert(0, str(Path(Block_bAE_PARENT_DIR).resolve()))
    from train_gruencoder_transformerdecoder import LitBlockbAE_Transformer
except ImportError:
    # Raise ImportError instead of exit()
    raise ImportError("Cannot import 'LitBlockbAE_Transformer'. Please check Block_bAE path or installation.")

# --- RDKit and SELFIES Import ---
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, QED
    from rdkit.Chem.MolStandardize import rdMolStandardize
    sys.path.append(str(Path(Chem.RDConfig.RDContribDir) / 'SA_Score'))
    from sascorer import calculateScore as calculate_sa_score
    import selfies as sf
    RDKIT_AVAILABLE = True
except ImportError as e:
    RDKIT_AVAILABLE = False
    print(f"Warning: RDKit or SELFIES module failed to load, some features will be unavailable. Error: {e}")
if RDKIT_AVAILABLE: Chem.rdBase.DisableLog('rdApp.*')

def _vae_inference_worker_queue(device, model_ckpt_path, latent_codes_chunk, batch_size, task_type, result_queue):
    try:
        vae_wrapper = VAEWrapper(model_ckpt_path, device)
        if task_type == 'decode':
            results = vae_wrapper.decode_to_ids_single_gpu(latent_codes_chunk, batch_size)
        elif task_type == 'recon_loss':
            results = vae_wrapper.calculate_reconstruction_loss_single_gpu(latent_codes_chunk, batch_size)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        result_queue.put(results)
    except Exception as e:
        import traceback; traceback.print_exc()
        result_queue.put(None)

class VAEWrapper:
    def __init__(self, model_ckpt_path, device, latent_h5_path=None):
        self.device = device
        self.model_ckpt_path = model_ckpt_path
        self.latent_h5_path = latent_h5_path  # New field
        self.model = None
        self.lit_model = None
        self.hparams = None

    def _lazy_load_model(self):
        if self.model is None:
            self.lit_model = LitBlockbAE_Transformer.load_from_checkpoint(
                self.model_ckpt_path, map_location=self.device
            )
            self.model = self.lit_model._uncompiled_model
            self.model.to(self.device)
            self.model.eval()
            self.hparams = self.lit_model.hparams
            self.sos_id = self.hparams.get('sos_id')
            self.eos_id = self.hparams.get('eos_id')
            self.max_len = self.hparams.get('max_len')
            self.pad_id = self.hparams.get('pad_id')

    def calculate_reconstruction_loss(self, latent_codes_numpy, batch_size=8192):
        return self._parallel_executor(latent_codes_numpy, batch_size, task_type="recon_loss")
    
    def decode_to_ids(self, latent_codes_numpy, batch_size=8192):
        return self._parallel_executor(latent_codes_numpy, batch_size, task_type="decode")
    
    def _parallel_executor(self, latent_codes_numpy, batch_size, task_type):
        # Support both single GPU (str) and multiple GPUs (list)
        if isinstance(self.device, str):
            num_gpus = 1
            devices = [self.device]
        elif isinstance(self.device, (list, tuple)):
            num_gpus = len(self.device)
            devices = self.device
        else:
            raise ValueError("device must be str or list[str]")

        print(f"[INFO] Detected {num_gpus} GPU(s), using Queue-based multi-process decoding...")

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        processes = []

        data_chunks = np.array_split(latent_codes_numpy, num_gpus)

        for i in range(num_gpus):
            p = ctx.Process(
                target=_vae_inference_worker_queue,
                args=(devices[i], self.model_ckpt_path, data_chunks[i], batch_size, task_type, result_queue)
            )
            p.start()
            processes.append(p)

        results_list = []
        for _ in range(num_gpus):
            results_list.append(result_queue.get())

        for p in processes:
            p.join()

        return np.concatenate(results_list, axis=0)

    @torch.no_grad()
    def calculate_reconstruction_loss_single_gpu(self, latent_codes_numpy, batch_size=4096):
        """[Solution A Final] Efficient unsupervised reconstruction loss calculation on single GPU."""
        self._lazy_load_model()
        
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(latent_codes_numpy))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_losses = []

        for z_batch_tuple in tqdm(loader, desc=f"[{self.device}] Calculating Fast Recon Loss"):
            z_batch = z_batch_tuple[0].to(self.device, dtype=torch.float32)

            # Step 1: Greedy generation of complete sequence
            generated_seqs = self.model.greedy_decode(
                z_batch, self.sos_id, self.eos_id, self.max_len
            )

            # Step 2: Parallel evaluation
            trg = generated_seqs[:, :-1]
            trg_y = generated_seqs[:, 1:]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                memory = self.model.from_latent(z_batch).unsqueeze(1).expand(-1, trg.size(1), -1)
                trg_emb = self.model.embedding(trg) * math.sqrt(self.model.d_model)
                trg_emb = self.model.pos_encoder(trg_emb)
                causal_mask = self.model._generate_square_subsequent_mask(trg.size(1), self.device)

                output = self.model.transformer_decoder(tgt=trg_emb, memory=memory, tgt_mask=causal_mask)
                logits = self.model.fc_out(output)

            # Calculate NLL
            log_probs = F.log_softmax(logits, dim=-1)
            nll_per_token = -log_probs.gather(2, trg_y.unsqueeze(-1)).squeeze(-1)

            # --- Core fix: Use simpler, more robust mask ---
            is_eos = (trg_y == self.eos_id)
            eos_indices = torch.cumsum(is_eos, dim=1)
            mask_before_eos = (eos_indices == 0)
            mask_at_first_eos = is_eos & (eos_indices == 1)
            final_mask = (mask_before_eos | mask_at_first_eos) & (trg_y != self.pad_id)
            # --- Fix end ---
            
            masked_nll = nll_per_token * final_mask
            loss_per_sample = masked_nll.sum(dim=1)
            
            all_losses.append(loss_per_sample.cpu().to(torch.float32))
            
        return torch.cat(all_losses).numpy()

    @torch.no_grad()
    def decode_to_ids_single_gpu(self, latent_codes_numpy, batch_size=8192):
        self._lazy_load_model()
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(latent_codes_numpy))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        decoded_sequences = []
        for z_batch_tuple in tqdm(loader, desc=f"[{self.device}] Decoding"):
            z_tensor = z_batch_tuple[0].to(self.device, dtype=torch.float32)
            
            try:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    id_seqs = self.model.greedy_decode(
                        z_tensor, self.sos_id, self.eos_id, self.max_len
                    )
                id_seqs_cpu = id_seqs.cpu().numpy().astype(np.int16)
                current_batch_size, current_len = id_seqs_cpu.shape
                if current_len < self.max_len:
                    padding = np.full(
                        (current_batch_size, self.max_len - current_len),
                        self.pad_id, dtype=np.int16
                    )
                    id_seqs_padded = np.hstack([id_seqs_cpu, padding])
                else:
                    id_seqs_padded = id_seqs_cpu
                decoded_sequences.append(id_seqs_padded)
            except Exception as e:
                logging.error(f"Error decoding batch: {e}")
                # Create failed sequence (all pad_id)
                failed_seqs = np.full((z_tensor.shape[0], self.max_len), self.pad_id, dtype=np.int16)
                decoded_sequences.append(failed_seqs)

        return np.vstack(decoded_sequences)


class SMILESReconstructor:
    """SELFIES to SMILES reconstructor - using selfies library for conversion"""

    def __init__(self, vocab_path: str, n_jobs: int = -1, standardize_smiles: bool = True):
        """
        Initialize SMILES reconstructor

        Args:
            vocab_path: Path to vocabulary JSON file
            n_jobs: Number of parallel jobs
            standardize_smiles: Whether to standardize SMILES (remove chirality info, etc.)
        """
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        self.id_to_token = vocab['itos_lst']
        token_to_id = {t: i for i, t in enumerate(self.id_to_token)}
        self.sos_id = token_to_id.get('<sos>')
        self.eos_id = token_to_id.get('<eos>')
        self.n_jobs = n_jobs
        self.standardize_smiles = standardize_smiles

        logging.info(f"SMILESReconstructor initialized - Vocab size: {len(self.id_to_token)}, Standardize: {standardize_smiles}")

    def reconstruct(self, decoded_ids_padded: np.ndarray, iteration_num: int = 0,
                   batch_size: int = 50_000, start_idx: int = 0) -> List[Dict]:
        """
        Reconstruct ID sequences to SMILES

        Args:
            decoded_ids_padded: ID sequence array
            iteration_num: Iteration number (for naming failed samples)
            batch_size: Batch size
            start_idx: Start index (for global indexing)

        Returns:
            List of reconstruction results, each containing key, success, original_smiles, standardized_smiles
        """
        n_samples = len(decoded_ids_padded)
        if n_samples == 0:
            return []

        log = logging.getLogger(__name__)
        log.info(f"Starting reconstruction of {n_samples} ID sequences to SMILES...")

        # 1. Dynamically calculate number of tasks based on data size and desired batch_size
        num_tasks = math.ceil(n_samples / batch_size)

        # 2. Split indices and data
        indices = np.arange(start_idx, start_idx + n_samples)
        id_chunks = np.array_split(decoded_ids_padded, num_tasks)
        index_chunks = np.array_split(indices, num_tasks)

        tasks = [
            delayed(self._reconstruct_worker)(
                id_chunk, index_chunk, self.id_to_token, self.sos_id,
                self.eos_id, iteration_num, self.standardize_smiles
            )
            for id_chunk, index_chunk in zip(id_chunks, index_chunks)
        ]

        n_jobs = effective_n_jobs(self.n_jobs)
        log.info(f"Reconstructing SMILES using {len(tasks)} tasks on {n_jobs} worker processes")

        results_list = Parallel(n_jobs=n_jobs)(
            tqdm(tasks, desc=f"[CPU] Iter {iteration_num} Reconstructing SMILES")
        )

        # Merge results
        results = [item for sublist in results_list for item in sublist]

        # Statistics
        success_count = sum(1 for r in results if r['success'])
        fail_count = len(results) - success_count
        log.info(f"Reconstruction complete - Success: {success_count}, Failed: {fail_count} (Success rate: {success_count/len(results):.2%})")

        # Failure reason breakdown
        from collections import Counter
        fail_reasons = Counter(r.get('fail_reason', 'unknown') for r in results if not r['success'])
        if fail_reasons:
            log.info("Failure reason distribution:")
            for reason, cnt in fail_reasons.items():
                log.info(f"  {reason:15s} : {cnt} ({cnt/len(results):.2%})")

        fail_log_path = Path(f"failed_selfies_iter{iteration_num:02d}.log")
        with open(fail_log_path, "w") as f:
            for r in results:
                if not r['success']:
                    f.write(f"{r['key']}\t{r.get('fail_reason')}\t{r.get('selfies')}\n")
        log.info(f"Failed SELFIES saved to {fail_log_path}")

        return results

    @staticmethod
    def _reconstruct_worker(id_sequences_chunk: np.ndarray, global_indices_chunk: np.ndarray,
                           id_to_token: List[str], sos_id: int, eos_id: int,
                           iteration_num: int, standardize_smiles: bool) -> List[Dict]:
        """
        Worker process: Reconstruct ID sequences to SMILES
        """
        # Re-import necessary modules
        import selfies as sf
        from rdkit import Chem
        from rdkit.Chem.MolStandardize import rdMolStandardize

        # Create standardizers (if needed)
        normalizer = None
        uncharger = None
        if standardize_smiles:
            normalizer = rdMolStandardize.Normalizer()
            uncharger = rdMolStandardize.Uncharger()

        results = []
        for id_seq, global_idx in zip(id_sequences_chunk, global_indices_chunk):
            try:
                # ======================== Core fix area (Solution 1) ========================
                token_seq = []
                # Use explicit for loop to process sequence
                for tid_int in id_seq:
                    tid = int(tid_int) # Ensure integer type

                    # 1. First check termination condition!
                    if tid == eos_id:
                        break  # Stop immediately, no further tokens processed

                    # 2. Then check if it's other special tokens to skip
                    elif tid == sos_id or tid == -1:
                        continue

                    # 3. Check if token ID is within vocabulary range
                    elif tid >= len(id_to_token):
                        # If ID is out of range, can warn or skip
                        warnings.warn(f"Found token ID {tid} outside of vocab size {len(id_to_token)}. Skipping.")
                        continue

                    # 4. Check if token itself is <pad>
                    elif id_to_token[tid] == '<pad>':
                        continue

                    # 5. If it's a valid, non-terminating token, add it
                    else:
                        token_seq.append(id_to_token[tid])

                if not token_seq:
                    results.append({
                        'key': f"fail_emptytokens_iter{iteration_num:02d}_{global_idx}",
                        'success': False,
                        'fail_reason': "empty_tokens",
                        'original_smiles': None,
                        'standardized_smiles': None
                    })
                    continue
                
                # 2. SELFIES → SMILES
                try:
                    selfies_string = ''.join(token_seq)
                    original_smi = sf.decoder(selfies_string)
                except Exception:
                    results.append({
                        'key': f"fail_selfies_iter{iteration_num:02d}_{global_idx}",
                        'success': False,
                        'fail_reason': "selfies_decode",
                        'original_smiles': None,
                        'standardized_smiles': None,
                        'selfies': selfies_string
                    })
                    continue

                if not original_smi:
                    results.append({
                        'key': f"fail_selfies_empty_iter{iteration_num:02d}_{global_idx}",
                        'success': False,
                        'fail_reason': "selfies_empty",
                        'original_smiles': None,
                        'standardized_smiles': None
                    })
                    continue

                # 3. RDKit parsing
                mol = Chem.MolFromSmiles(original_smi)
                if mol is None:
                    results.append({
                        'key': f"fail_rdkit_parse_iter{iteration_num:02d}_{global_idx}",
                        'success': False,
                        'fail_reason': "rdkit_parse",
                        'original_smiles': original_smi,
                        'standardized_smiles': None
                    })
                    continue

                # 4. Standardization
                final_smiles, standardized_smiles = original_smi, None
                if standardize_smiles:
                    try:
                        Chem.RemoveStereochemistry(mol)
                        mol = normalizer.normalize(mol)
                        mol = uncharger.uncharge(mol)
                        standardized_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
                        if standardized_smiles:
                            final_smiles = standardized_smiles
                    except Exception:
                        results.append({
                            'key': f"fail_standardize_iter{iteration_num:02d}_{global_idx}",
                            'success': False,
                            'fail_reason': "standardize",
                            'original_smiles': original_smi,
                            'standardized_smiles': None
                        })
                        continue

                # Success
                results.append({
                    'key': final_smiles,
                    'success': True,
                    'fail_reason': None,
                    'original_smiles': original_smi,
                    'standardized_smiles': standardized_smiles
                })
            
            except Exception:
                results.append({
                    'key': f"fail_unknown_iter{iteration_num:02d}_{global_idx}",
                    'success': False,
                    'fail_reason': "unknown",
                    'original_smiles': None,
                    'standardized_smiles': None
                })
        
        return results

    def reconstruct_simple(self, decoded_ids_padded: np.ndarray,
                          prefix: str = "fail") -> List[str]:
        """
        Simplified reconstruction method, returns only SMILES string list

        Args:
            decoded_ids_padded: ID sequence array
            prefix: Prefix for failed samples

        Returns:
            SMILES string list
        """
        results = self.reconstruct(decoded_ids_padded)
        return [result['key'] for result in results]

def _calculate_worker_from_file(h5_path, indices, prop_funcs, penalty_values):
    """Worker: Read data from HDF5 file by index to avoid memory explosion"""
    results = {}
    with h5py.File(h5_path, 'r') as hf:
        smiles_dset = hf['smiles']
        smiles_chunk = [s.decode('utf-8') for s in smiles_dset[indices]]

    for smi in smiles_chunk:
        penalty = dict(penalty_values)
        if smi.startswith("fail_"):
            results[smi] = penalty
            continue
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                results[smi] = penalty
                continue
            props = {name: func(mol) for name, func in prop_funcs.items()}
            violations = sum([
                props['mw'] >= 500,
                props['logp'] > 5,
                props['hbd'] > 5,
                props['hba'] > 10
            ])
            props['lipinski_violations'] = violations
            props['passes_lipinski_ro5'] = int(violations == 0)
            results[smi] = props
        except Exception:
            results[smi] = penalty
    return results


class PropertyCalculator:
    PROPERTY_FUNCTIONS = {'qed': QED.qed, 'sa_score': calculate_sa_score, 'logp': Crippen.MolLogP, 'mw': Descriptors.ExactMolWt, 'tpsa': Descriptors.TPSA, 'nrb': Descriptors.NumRotatableBonds, 'hbd': Descriptors.NumHDonors, 'hba': Descriptors.NumHAcceptors}
    PENALTY_VALUES = {'qed': 0.0, 'sa_score': 10.0, 'logp': -10.0, 'mw': 1000.0, 'tpsa': 0.0, 'nrb': 50, 'hbd': 0, 'hba': 0, 'lipinski_violations': 5, 'passes_lipinski_ro5': 0}
    
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs

    def calculate(self, smiles_list, batch_size=50_000):
        """
        Parallel calculation of RDKit properties for SMILES using joblib.
        """
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit is not available for property calculation.")

        n_samples = len(smiles_list)
        if n_samples == 0: return {}
        log = logging.getLogger(__name__)

        # 1. Dynamically calculate number of tasks
        num_tasks = math.ceil(n_samples / batch_size)

        # 2. Split data
        smiles_chunks = np.array_split(smiles_list, num_tasks)

        tasks = [
            delayed(self._calculate_worker)(chunk, self.PROPERTY_FUNCTIONS, self.PENALTY_VALUES)
            for chunk in smiles_chunks
        ]

        n_jobs = effective_n_jobs(self.n_jobs)
        log.info(f"Calculating properties for {n_samples} SMILES with {len(tasks)} tasks on {n_jobs} workers.")

        results_list = Parallel(n_jobs=n_jobs)(
            tqdm(tasks, desc="[CPU] Calculating RDKit properties")
        )

        final_props = {k: v for d in results_list for k, v in d.items()}
        return final_props

    @staticmethod
    def _calculate_worker(smiles_chunk, prop_funcs, penalty_values):
        results = {}
        for smi in smiles_chunk:
            # Return a copy of penalty_values
            penalty = dict(penalty_values)
            if smi.startswith("fail_"):
                results[smi] = penalty
                continue
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    results[smi] = penalty
                    continue
                props = {name: func(mol) for name, func in prop_funcs.items()}
                violations = sum([props['mw'] >= 500, props['logp'] > 5, props['hbd'] > 5, props['hba'] > 10])
                props['lipinski_violations'] = violations
                props['passes_lipinski_ro5'] = int(violations <= 1)
                results[smi] = props
            except Exception:
                results[smi] = penalty
        return results
    
    def calculate_from_h5(self, h5_path, batch_size=50_000):
        """
        Calculate SMILES properties directly from HDF5 file in chunks (recommended for very large datasets)
        """
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit is not available for property calculation.")

        log = logging.getLogger(__name__)
        with h5py.File(h5_path, 'r') as hf:
            n_samples = len(hf['smiles'])

        if n_samples == 0:
            return {}

        # Chunk by index rather than loading entire smiles_list
        all_indices = np.arange(n_samples)
        num_tasks = math.ceil(n_samples / batch_size)
        index_chunks = np.array_split(all_indices, num_tasks)

        tasks = [
            delayed(_calculate_worker_from_file)(
                h5_path, chunk, self.PROPERTY_FUNCTIONS, self.PENALTY_VALUES
            )
            for chunk in index_chunks
        ]

        n_jobs = effective_n_jobs(self.n_jobs)
        log.info(f"Calculating properties for {n_samples:,} SMILES "
                f"with {len(tasks)} tasks on {n_jobs} workers.")

        results_list = Parallel(n_jobs=n_jobs)(
            tqdm(tasks, desc="[CPU] Calculating RDKit properties")
        )

        final_props = {k: v for d in results_list for k, v in d.items()}
        return final_props
    

class ScoreCalculator:
    # V6 weights: Clearly separate base score and loss term weights
    BASE_SCORE_WEIGHT = 0.9  # Property score accounts for 90%
    LOSS_TERM_WEIGHT = 0.1   # Loss term accounts for 10%

    # Internal weights for property score, sum to 1.0
    PROPERTY_WEIGHTS = {
        'qed': 0.30, 'sa_score': 0.20, 'logp': 0.10, 'tpsa': 0.10,
        'lipinski_pass': 0.10, 'mw': 0.10, 'nrb': 0.10,
    }
    assert np.isclose(sum(PROPERTY_WEIGHTS.values()), 1.0), "Property weights must sum to 1.0"

    # Reconstruction loss penalty factor
    RECON_LOSS_PENALTY_FACTOR = 0.5

    # Fixed normalization parameters
    # WARNING: These values are based on old V3 score calculation.
    # After applying V6 logic, you must re-run score_processor.py once
    # to calculate new global mean and std, then return to update these two values!
    FIXED_SCORE_MEAN = 0.5612 # Placeholder, update with score_processor.py
    FIXED_SCORE_STD = 0.3302  # Placeholder, update with score_processor.py

    @staticmethod
    def _gaussian_term(x, mu, sigma):
        return np.exp(-np.square(x - mu) / (2 * np.square(sigma)))

    def calculate_raw_scores(self, data_dict: dict) -> np.ndarray:
        """
        [V6 Final] Score calculation logic
        - Unified scale, weighted sum of property base score and reconstruction loss adjustment term.
        - This method is now fully consistent with logic in score_processor.py.
        """
        n_samples = len(data_dict['keys'])

        # --- Step 1: Calculate base property score (Base_Score) ---
        base_scores = np.zeros(n_samples, dtype=np.float32)

        base_scores += self.PROPERTY_WEIGHTS['qed'] * data_dict['qed']
        base_scores += self.PROPERTY_WEIGHTS['sa_score'] * np.maximum(0, (10 - data_dict['sa_score']) / 9.0)
        base_scores += self.PROPERTY_WEIGHTS['logp'] * self._gaussian_term(data_dict['logp'], mu=2.5, sigma=1.5)
        base_scores += self.PROPERTY_WEIGHTS['tpsa'] * self._gaussian_term(data_dict['tpsa'], mu=80, sigma=30)
        base_scores += self.PROPERTY_WEIGHTS['mw'] * self._gaussian_term(data_dict['mw'], mu=350, sigma=100)
        base_scores += self.PROPERTY_WEIGHTS['nrb'] * self._gaussian_term(data_dict['nrb'], mu=4, sigma=2)
        base_scores += self.PROPERTY_WEIGHTS['lipinski_pass'] * data_dict['passes_lipinski_ro5'].astype(np.float32)

        # --- Step 2: Calculate reconstruction loss adjustment term (Loss_Term) ---
        loss_term = np.exp(-self.RECON_LOSS_PENALTY_FACTOR * data_dict['reconstruction_loss'])

        # --- Step 3: Weighted sum for final score ---
        final_scores = (self.BASE_SCORE_WEIGHT * base_scores) + (self.LOSS_TERM_WEIGHT * loss_term)

        return final_scores.astype(np.float32)

    def standardize_scores(self, raw_scores: np.ndarray) -> np.ndarray:
        """Standardize raw scores using fixed global parameters."""
        if self.FIXED_SCORE_STD > 1e-6:
            return ((raw_scores - self.FIXED_SCORE_MEAN) / self.FIXED_SCORE_STD).astype(np.float32)

        # Edge case prevention for division by zero
        return (raw_scores - self.FIXED_SCORE_MEAN).astype(np.float32)


# --- Convenience functions: Simplified calls ---

def create_vae_wrapper(model_ckpt_path: str, device: str = "cuda:0", latent_h5_path: str = None) -> VAEWrapper:
    """Convenience function to create VAE wrapper"""
    return VAEWrapper(model_ckpt_path, device, latent_h5_path)

def create_smiles_reconstructor(vocab_path: str, n_jobs: int = -1,
                               standardize_smiles: bool = True) -> SMILESReconstructor:
    """Convenience function to create SMILES reconstructor"""
    return SMILESReconstructor(vocab_path, n_jobs, standardize_smiles)

def create_property_calculator(n_jobs: int = -1) -> PropertyCalculator:
    """Convenience function to create property calculator"""
    return PropertyCalculator(n_jobs)

def create_score_calculator() -> ScoreCalculator:
    """Convenience function to create score calculator"""
    return ScoreCalculator()

def decode_and_reconstruct(vae_wrapper: VAEWrapper, reconstructor: SMILESReconstructor,
                          latent_codes: np.ndarray, decode_batch_size: int = 4096,
                          reconstruct_batch_size: int = 50000) -> List[Dict]:
    """
    One-stop decode and reconstruct function

    Args:
        vae_wrapper: VAE wrapper instance
        reconstructor: SMILES reconstructor instance
        latent_codes: Latent vector array
        decode_batch_size: Decode batch size
        reconstruct_batch_size: Reconstruct batch size

    Returns:
        Reconstruction result list
    """
    # Decode latent vectors to ID sequences
    decoded_ids = vae_wrapper.decode_to_ids(latent_codes, batch_size=decode_batch_size)

    # Reconstruct ID sequences to SMILES
    results = reconstructor.reconstruct(decoded_ids, batch_size=reconstruct_batch_size)

    return results