import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar
from pytorch_lightning.strategies import DDPStrategy
from swanlab.integration.pytorch_lightning import SwanLabLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ===============================================================
# 1. RBM Model Definition
# ===============================================================
class RBM(nn.Module):
    """Restricted Boltzmann Machine"""
    
    def __init__(self, n_visible, n_hidden):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        
        # Weights and biases
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        
    def sample_h(self, v):
        """Sample hidden layer from visible layer (v -> h)"""
        h_prob = torch.sigmoid(F.linear(v, self.W.t(), self.h_bias))
        h_sample = torch.bernoulli(h_prob)
        return h_prob, h_sample
    
    def sample_v(self, h):
        """Sample visible layer from hidden layer (h -> v)"""
        v_prob = torch.sigmoid(F.linear(h, self.W, self.v_bias))
        v_sample = torch.bernoulli(v_prob)
        return v_prob, v_sample
    
    def free_energy(self, v):
        """Compute free energy"""
        vbias_term = torch.matmul(v, self.v_bias)
        wx_b = F.linear(v, self.W.t(), self.h_bias)
        hidden_term = torch.sum(F.softplus(wx_b), dim=1)
        return -vbias_term - hidden_term
    
    def forward(self, v, k=1):
        """
        Contrastive Divergence CD-k algorithm
        Args:
            v: Input visible layer data
            k: Number of Gibbs sampling steps
        Returns:
            v_recon: Reconstructed visible layer
            h_prob: Hidden layer probabilities
        """
        # Positive phase: obtain hidden layer from data
        h_prob, h_sample = self.sample_h(v)

        # Gibbs sampling for k steps
        chain_v = v
        for _ in range(k):
            _, h_gibbs = self.sample_h(chain_v)
            _, chain_v = self.sample_v(h_gibbs)

        return chain_v, h_prob

# ===============================================================
# 2. Efficient In-Memory Dataset
# ===============================================================
class InMemoryDataset(Dataset):
    """In-memory dataset to avoid additional overhead"""

    def __init__(self, X: torch.Tensor):
        self.X = X.float()  # RBM requires float type
    
    def __getitem__(self, idx):
        return self.X[idx]
    
    def __len__(self):
        return len(self.X)

def load_h5_dataset(h5_path: str, valid_size: float = 0.01, seed: int = 42):
    """Load data from H5 file and split into train/validation sets"""
    log.info(f"Loading dataset from {h5_path}...")
    
    with h5py.File(h5_path, 'r') as hf:
        X_all = torch.from_numpy(hf['latent_codes'][:]).float().contiguous()
    
    log.info(f"Dataset loaded. Total samples: {len(X_all)}, Feature dim: {X_all.shape[1]}")

    # Split dataset
    train_indices, val_indices = train_test_split(
        np.arange(len(X_all)), test_size=valid_size, random_state=seed
    )
    
    X_train = X_all[train_indices].contiguous()
    X_val = X_all[val_indices].contiguous()
    
    train_dataset = InMemoryDataset(X_train)
    val_dataset = InMemoryDataset(X_val)
    
    del X_all
    import gc; gc.collect()
    
    log.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    return train_dataset, val_dataset

# ===============================================================
# 3. PyTorch Lightning Module
# ===============================================================
class LitRBM(pl.LightningModule):
    """PyTorch Lightning wrapper for RBM"""
    
    def __init__(self, n_visible, n_hidden, learning_rate=1e-3, 
                 weight_decay=1e-5, cd_k=1):
        super().__init__()
        self.save_hyperparameters()
        
        self.rbm = RBM(n_visible, n_hidden)
        self.cd_k = cd_k
        
    def training_step(self, batch, batch_idx):
        v_real = batch

        # CD-k training
        v_recon, _ = self.rbm(v_real, k=self.cd_k)

        # Compute loss: free energy difference
        loss = torch.mean(self.rbm.free_energy(v_real) - self.rbm.free_energy(v_recon))

        # Compute reconstruction error (for monitoring)
        recon_error = F.mse_loss(v_recon, v_real)
        
        self.log('train/loss', loss, prog_bar=True, sync_dist=True)
        self.log('train/recon_error', recon_error, prog_bar=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        v_real = batch

        # Use CD-1 for validation
        v_recon, h_prob = self.rbm(v_real, k=1)

        # Compute metrics
        recon_error = F.mse_loss(v_recon, v_real)
        loss = torch.mean(self.rbm.free_energy(v_real) - self.rbm.free_energy(v_recon))

        # Compute hidden layer activation
        h_activation = h_prob.mean()
        active_bits = (h_prob > 0.5).float().sum(dim=1).mean()
        
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        self.log('val/recon_error', recon_error, prog_bar=True, sync_dist=True)
        self.log('val/hidden_activation', h_activation, sync_dist=True)
        self.log('val/active_bits', active_bits, sync_dist=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.rbm.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        # Use cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs,
            eta_min=self.hparams.learning_rate * 0.1
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }

# ===============================================================
# 4. Data Module
# ===============================================================
class RBMDataModule(pl.LightningDataModule):
    """RBM data module"""
    
    def __init__(self, train_dataset, val_dataset, batch_size=512, num_workers=8):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

# ===============================================================
# 5. Main Training Function
# ===============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train RBM on binary latent codes")

    # Data parameters
    parser.add_argument("--h5_path", type=str, required=True, help="Path to H5 file")
    parser.add_argument("--valid_size", type=float, default=0.005, help="Validation set ratio")

    # Model parameters
    parser.add_argument("--n_hidden", type=int, default=256, help="Number of hidden units")
    parser.add_argument("--cd_k", type=int, default=1, help="k value in CD-k")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32768, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum training epochs")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loading workers")

    # Hardware parameters
    parser.add_argument("--devices", type=int, default=2, help="Number of GPUs")
    parser.add_argument("--accelerator", type=str, default="gpu", help="Accelerator type")

    # Logging parameters
    parser.add_argument("--save_dir", type=str, default="./rbm_outputs", help="Save directory")
    parser.add_argument("--swanlab_project", type=str, default="RBM-Training", help="SwanLab project name")
    parser.add_argument("--experiment_name", type=str, default="rbm_experiment", help="Experiment name")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    
    return parser.parse_args()

def main():
    args = parse_args()

    # Set random seed
    pl.seed_everything(args.seed, workers=True)

    # Auto-detect feature dimension
    with h5py.File(args.h5_path, 'r') as hf:
        n_visible = hf['latent_codes'].shape[1]
    log.info(f"Auto-detected n_visible = {n_visible}")

    # Load data
    train_dataset, val_dataset = load_h5_dataset(
        args.h5_path, 
        valid_size=args.valid_size, 
        seed=args.seed
    )
    
    # Create data module
    dm = RBMDataModule(
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Create model
    model = LitRBM(
        n_visible=n_visible,
        n_hidden=args.n_hidden,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        cd_k=args.cd_k
    )

    # Create logger
    swanlab_logger = SwanLabLogger(
        project=args.swanlab_project,
        experiment_name=args.experiment_name,
        save_dir=args.save_dir
    )

    # Setup callbacks
    checkpoint_dir = Path(args.save_dir) / "checkpoints"
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor='val/recon_error',
            mode='min',
            save_top_k=3,
            filename='rbm-{epoch:02d}-{val/recon_error:.4f}'
        ),
        EarlyStopping(
            monitor='val/recon_error',
            patience=args.patience,
            mode='min'
        ),
        LearningRateMonitor(logging_interval='epoch'),
        RichProgressBar()
    ]
    
    # Configure strategy
    if args.devices > 1:
        strategy = DDPStrategy(
            find_unused_parameters=False,
            static_graph=True,
            gradient_as_bucket_view=True
        )
    else:
        strategy = "auto"

    # Create trainer
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=swanlab_logger,
        strategy=strategy,
        precision="32",  # RBM is typically more stable with FP32
        gradient_clip_val=1.0,
        log_every_n_steps=10
    )

    # Start training
    log.info(f"Starting RBM training for {args.max_epochs} epochs...")
    log.info(f"Model: {n_visible} visible -> {args.n_hidden} hidden units")
    log.info(f"Using CD-{args.cd_k} algorithm")

    trainer.fit(model, dm)

    # Output best model path
    best_model_path = callbacks[0].best_model_path
    log.info(f"Training finished! Best checkpoint: {best_model_path}")

if __name__ == "__main__":
    main()