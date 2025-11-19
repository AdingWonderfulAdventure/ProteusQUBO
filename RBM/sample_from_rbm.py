import argparse
import torch
import h5py
from pathlib import Path
import logging
from tqdm import tqdm

# ===============================================================
# 1. Import RBM Model Definition
# ===============================================================
from train_rbm import RBM, LitRBM 

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ===============================================================
# 2. Modified Sampling Function (Supports Batching)
# ===============================================================
@torch.no_grad()
def generate_samples_batched(
    rbm_model: RBM, 
    n_samples_total: int, 
    batch_size: int,
    n_gibbs_steps: int, 
    burn_in: int, 
    device: str
) -> torch.Tensor:
    """
    Generate new samples from a trained RBM using batched Gibbs sampling to avoid GPU memory overflow.
    """
    rbm_model.to(device)
    rbm_model.eval()

    all_samples = []
    n_batches = (n_samples_total + batch_size - 1) // batch_size
    
    log.info(f"Starting batched Gibbs sampling for {n_samples_total} total samples in {n_batches} batches...")

    for i in tqdm(range(n_batches), desc="Generating Batches"):
        current_batch_size = min(batch_size, n_samples_total - i * batch_size)

        # 1. Randomly initialize visible layer vectors for current batch
        v = torch.randint(0, 2, (current_batch_size, rbm_model.n_visible), dtype=torch.float, device=device)

        # 2. Run Gibbs sampling chain (burn-in + sampling)
        # For efficiency, we can run chains in parallel across all batches
        # Burn-in period
        for _ in range(burn_in):
            _, h = rbm_model.sample_h(v)
            _, v = rbm_model.sample_v(h)

        # Final sampling steps
        for _ in range(n_gibbs_steps - burn_in):
            _, h = rbm_model.sample_h(v)
            _, v = rbm_model.sample_v(h)

        # 3. Move batch results to CPU and store
        all_samples.append(v.cpu())

    log.info("All batches generated. Concatenating results...")
    return torch.cat(all_samples, dim=0)

# ===============================================================
# 3. Main Execution Function
# ===============================================================
def main():
    parser = argparse.ArgumentParser(description="Generate new samples from a trained RBM model using batched sampling.")

    # --- Core parameters ---
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/root/workspace/ProteusQUBO/RBM/rbm_outputs/checkpoints/rbm-epoch=35-val/recon_error=0.0617.ckpt",
        help="Path to the trained PyTorch Lightning checkpoint (.ckpt file)."
    )
    parser.add_argument(
        "--output_h5",
        type=str,
        default="/root/workspace/ProteusQUBO/RBM/all.h5",
        help="Path to save the generated H5 file."
    )

    # --- Sampling parameters ---
    parser.add_argument("--n_samples", type=int, default=20000000, help="Total number of latent codes to generate.")
    parser.add_argument("--batch_size", type=int, default=262144, help="Batch size for sampling to control memory usage.")
    parser.add_argument("--n_gibbs_steps", type=int, default=2000, help="Total number of Gibbs sampling steps.")
    parser.add_argument("--burn_in", type=int, default=1000, help="Number of burn-in steps to discard.")

    # --- Model architecture parameters ---
    parser.add_argument("--n_visible", type=int, default=128, help="Number of visible units (feature dimension).")
    parser.add_argument("--n_hidden", type=int, default=256, help="Number of hidden units used during training.")
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f"Using device: {device}")

    try:
        log.info(f"Loading model from checkpoint: {args.checkpoint_path}")
        lit_model = LitRBM.load_from_checkpoint(
            args.checkpoint_path,
            n_visible=args.n_visible,
            n_hidden=args.n_hidden
        )
        rbm_model = lit_model.rbm

        # Call batched sampling function
        generated_samples = generate_samples_batched(
            rbm_model=rbm_model,
            n_samples_total=args.n_samples,
            batch_size=args.batch_size,
            n_gibbs_steps=args.n_gibbs_steps,
            burn_in=args.burn_in,
            device=device
        )
        
        log.info(f"Successfully generated {generated_samples.shape[0]} samples of dimension {generated_samples.shape[1]}.")

        output_path = Path(args.output_h5)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_path, 'w') as hf:
            hf.create_dataset('latent_codes', data=generated_samples.numpy(), compression="gzip")

        log.info(f"Samples saved successfully to: {output_path}")

    except Exception as e:
        log.error(f"An error occurred: {e}", exc_info=True)  # exc_info=True prints full traceback
        log.error("Please ensure the model architecture parameters (n_visible, n_hidden) match the checkpoint.")

if __name__ == "__main__":
    main()