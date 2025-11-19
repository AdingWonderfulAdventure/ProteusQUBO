import argparse
import h5py
import numpy as np
import logging
from pathlib import Path
import sys

# Import utils_Qmol_FM
utils_Qmol_FM_DIR = "/root/workspace/ProteusQUBO/Qmol_FM"
sys.path.insert(0, str(Path(utils_Qmol_FM_DIR).resolve()))
from utils_Qmol_FM import (
    create_vae_wrapper, 
    create_smiles_reconstructor,
    decode_and_reconstruct
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def latent_to_smiles(latent_codes: np.ndarray,
                     model_ckpt_path: str,
                     vocab_path: str,
                     device: str = "cuda:0",
                     decode_batch_size: int = 8192,
                     reconstruct_batch_size: int = 20000):
    """
    Convert latent_codes to SMILES strings
    """
    log.info(f"Loading VAE model: {model_ckpt_path}")
    vae_wrapper = create_vae_wrapper(model_ckpt_path, device, None)

    log.info(f"Loading vocabulary: {vocab_path}")
    reconstructor = create_smiles_reconstructor(vocab_path, n_jobs=-1, standardize_smiles=True)

    log.info(f"Starting decoding latent_codes → SMILES, total {len(latent_codes)} samples")
    conversion_results = decode_and_reconstruct(
        vae_wrapper=vae_wrapper,
        reconstructor=reconstructor,
        latent_codes=latent_codes,
        decode_batch_size=decode_batch_size,
        reconstruct_batch_size=reconstruct_batch_size
    )
    return conversion_results

def main(args):
    # Read latent_codes and labels
    log.info(f"Reading latent_codes and labels from {args.input_h5}")
    with h5py.File(args.input_h5, "r") as hf:
        latent_codes = hf["latent_codes"][:, :128]

        # Robustly read labels
        if "labels" in hf:
            log.info("Found 'labels' dataset, reading...")
            raw_labels = hf["labels"][:]
            # Handle byte strings or numeric labels, ensure we get a list of strings
            labels_list = [
                label.decode('utf-8') if isinstance(label, bytes) else str(label)
                for label in raw_labels
            ]
        else:
            log.warning("'labels' dataset not found in input file, generating default labels.")
            labels_list = [f"label_{i}" for i in range(len(latent_codes))]
    # =================================================================

    # Conversion
    conversion_results = latent_to_smiles(
        latent_codes=latent_codes,
        model_ckpt_path=args.model_ckpt_path,
        vocab_path=args.vocab_path,
        device=args.device,
        decode_batch_size=args.decode_batch_size,
        reconstruct_batch_size=args.reconstruct_batch_size
    )

    # Save HDF5
    output_h5_path = Path(args.output_h5)
    output_h5_path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Saving SMILES to HDF5: {output_h5_path}")

    # Extract SMILES and related information
    smiles_list = [res.get("standardized_smiles") or res.get("original_smiles") or "fail" for res in conversion_results]
    smiles_bytes = [s.encode("utf-8") for s in smiles_list]

    # Create HDF5 file and save
    with h5py.File(output_h5_path, "w") as hf_out:
        # Save SMILES data
        hf_out.create_dataset("smiles", data=smiles_bytes, compression="gzip")

        # Can also save latent_codes if needed
        hf_out.create_dataset("latent_codes", data=latent_codes, compression="gzip")

        if labels_list:
             hf_out.create_dataset("labels", data=labels_list,  compression="gzip")

        hf_out.attrs["total_samples"] = len(smiles_list)
        hf_out.attrs["successful_samples"] = sum(1 for res in conversion_results if res.get('success', False))
        hf_out.attrs["failed_samples"] = len(smiles_list) - hf_out.attrs["successful_samples"]

    log.info("✅ Conversion completed and saved successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert latent_codes to SMILES and save (TXT + HDF5)")
    parser.add_argument("--input_h5", type=str, required=True, help="Input HDF5 file containing latent_codes dataset")
    parser.add_argument("--model_ckpt_path", type=str, required=True, help="VAE model checkpoint path")
    parser.add_argument("--vocab_path", type=str, required=True, help="Vocabulary JSON file path")
    parser.add_argument("--output_h5", type=str, required=True, help="Output SMILES HDF5 file path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Computing device (default: cuda:0)")
    parser.add_argument("--decode_batch_size", type=int, default=8192, help="VAE decoding batch size")
    parser.add_argument("--reconstruct_batch_size", type=int, default=2000, help="SMILES conversion batch size")
    args = parser.parse_args()

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    main(args)
