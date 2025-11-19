import torch
import argparse
from pathlib import Path
import h5py
from tqdm import tqdm
import logging
import numpy as np
import json
import pandas as pd # Import Pandas for efficient TSV processing

# --- Ensure BlockCollator is imported ---
# Assume these files are in the same project as your script
from datamodule import BlockCollator 
from train_gruencoder_transformerdecoder import LitBlockbAE_Transformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# --- Add a custom dataset class to handle Token ID lists in memory ---
class InMemoryTokenDataset(torch.utils.data.Dataset):
    def __init__(self, token_ids_list):
        self.token_ids_list = token_ids_list

    def __len__(self):
        return len(self.token_ids_list)

    def __getitem__(self, idx):
        # Directly return token id sequence that's already a tensor
        return torch.tensor(self.token_ids_list[idx], dtype=torch.long)

def main(args):
    # --- Set up environment ---
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ==============================================================================
    # ❗️ Core modification #1: Read, deduplicate and process data from TSV file
    # ==============================================================================
    log.info(f"Reading data from TSV file: {args.tsv_path}")
    # Use Pandas for efficient reading, only read columns we need
    df = pd.read_csv(
        args.tsv_path,
        sep='\t',
        usecols=['smiles_no_isomers', 'token_ids'],
        on_bad_lines='skip' # Skip malformed lines
    )

    # Sanity check: ensure 'token_ids' column exists and has no missing values
    if 'token_ids' not in df.columns:
        raise ValueError("Fatal error: 'token_ids' column not found in TSV file.")

    if 'smiles_no_isomers' not in df.columns:
        raise ValueError("Fatal error: 'smiles_no_isomers' column not found in TSV file.")

    # Remove rows with missing values in either column
    initial_len = len(df)
    df.dropna(subset=['token_ids', 'smiles_no_isomers'], inplace=True)
    log.info(f"Read {initial_len} rows from TSV file, {len(df)} valid rows remaining after removing missing values.")

    log.info("Deduplicating based on token_ids...")
    # --- ✨ Key modification: Change subset from 'smiles_no_isomers' to 'token_ids' ---
    df.drop_duplicates(subset=['token_ids'], keep='first', inplace=True)
    log.info(f"{len(df)} unique token_ids sequences remaining after deduplication.")

    # Extract deduplicated SMILES and token_ids
    # At this point, SMILES correspond to unique token_ids (keeping the first one)
    unique_smiles_order = df['smiles_no_isomers'].tolist()

    log.info("Converting token_ids strings to integer lists...")
    # This process might be slow, so add tqdm progress bar
    try:
        token_ids_list = [
            [int(token) for token in row.split()]
            for row in tqdm(df['token_ids'], desc="Parsing token IDs")
        ]
    except ValueError as e:
        log.error(f"Error parsing token_ids: {e}")
        raise ValueError("token_ids column contains values that cannot be converted to integers")

    # Free memory occupied by Pandas DataFrame
    del df
    import gc
    gc.collect()

    # Check for empty token_ids sequences
    empty_sequences = [i for i, seq in enumerate(token_ids_list) if len(seq) == 0]
    if empty_sequences:
        log.warning(f"Found {len(empty_sequences)} empty token_ids sequences at positions: {empty_sequences[:10]}...")
        # Can choose to remove empty sequences or raise error
        # Here we choose to remove empty sequences
        token_ids_list = [seq for seq in token_ids_list if len(seq) > 0]
        unique_smiles_order = [smile for i, smile in enumerate(unique_smiles_order)
                              if i not in empty_sequences]
        log.info(f"{len(token_ids_list)} valid sequences remaining after removing empty sequences")

    # --- Load your trained model (unchanged) ---
    log.info(f"Loading model from checkpoint: {args.trained_model_ckpt_path}")
    try:
        lit_model = LitBlockbAE_Transformer.load_from_checkpoint(
            checkpoint_path=args.trained_model_ckpt_path,
            map_location=device
        )
    except Exception as e:
        log.error(f"Error loading model checkpoint: {e}")
        raise

    model = lit_model.model
    model.to(device)
    model.eval()

    # ==============================================================================
    # ❗️ Core modification #2: Get pad_id directly from model hparams and create DataLoader
    # ==============================================================================
    log.info("Getting hparams directly from loaded model...")
    if not hasattr(lit_model, 'hparams'):
        raise AttributeError("Model does not have 'hparams' attribute")

    if 'pad_id' not in lit_model.hparams:
        raise KeyError("Fatal error: 'pad_id' not found in model hyperparameters.")

    pad_id = lit_model.hparams.pad_id
    log.info(f"Got pad_id from model hparams: {pad_id}")

    # Check max_len parameter
    if 'max_len' not in lit_model.hparams:
        log.warning("'max_len' not found in model hparams, using default value 512")
        max_len = 512
    else:
        max_len = lit_model.hparams.max_len
    log.info(f"Model maximum sequence length: {max_len}")

    collator = BlockCollator(pad_id=pad_id)

    # Use our new in-memory dataset class
    inference_dataset = InMemoryTokenDataset(token_ids_list)

    data_loader = torch.utils.data.DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        shuffle=False, # Must be False to maintain order
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # --- Perform inference (logic unchanged) ---
    all_latent_codes = []
    log.info(f"Starting inference, total {len(inference_dataset)} samples, batch size {args.batch_size}...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Encoding sequences")):
            try:
                src = batch['src'].to(device)

                # --- Key: Ensure input sequence length doesn't exceed model's maximum supported length ---
                if src.shape[1] > max_len:
                    src = src[:, :max_len]
                    log.debug(f"Batch {batch_idx}: Truncating sequence length from {src.shape[1]} to {max_len}")

                binary_codes = model.get_deterministic_binary_representation(src)
                all_latent_codes.append(binary_codes.cpu())

            except Exception as e:
                log.error(f"Error processing batch {batch_idx}: {e}")
                raise

    # --- Consolidate results and save (logic unchanged) ---
    log.info("Inference complete. Consolidating results...")
    final_codes_tensor = torch.cat(all_latent_codes, dim=0)
    final_codes_np = final_codes_tensor.numpy().astype(np.uint8) # Using uint8 is more standard

    # Verify result length consistency
    if len(final_codes_np) != len(unique_smiles_order):
        raise ValueError(f"Result length mismatch: latent codes {len(final_codes_np)} vs SMILES {len(unique_smiles_order)}")

    log.info(f"Saving {len(unique_smiles_order)} SMILES and their corresponding latent vectors to: {output_path}")
    try:
        with h5py.File(output_path, 'w') as hf:
            smiles_bytes = [s.encode('utf-8') for s in unique_smiles_order]
            hf.create_dataset('smiles', data=smiles_bytes, compression="gzip")
            hf.create_dataset('latent_codes', data=final_codes_np, compression="gzip")

            # Add metadata
            hf.attrs['num_samples'] = len(unique_smiles_order)
            hf.attrs['latent_dim'] = final_codes_np.shape[1] if len(final_codes_np.shape) > 1 else 1
            hf.attrs['deduplication_method'] = 'token_ids'
    except Exception as e:
        log.error(f"Error saving HDF5 file: {e}")
        raise

    log.info("Processing completed successfully.")
    log.info(f"Final statistics: Processed {len(unique_smiles_order)} unique token_ids sequences")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Read TSV file, deduplicate SMILES and Token IDs, run inference, and save aligned SMILES-Latent HDF5 file.")

    # --- Core modification #3: Update command line arguments ---
    parser.add_argument("--tsv_path", type=str, required=True, help="Path to .tsv file containing SMILES and Token IDs.")
    parser.add_argument("--trained_model_ckpt_path", type=str, required=True, help="Path to trained model checkpoint (.ckpt) file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save output HDF5 file.")

    parser.add_argument("--batch_size", type=int, default=8192, help="Batch size for inference.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker processes for data loader.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for inference.")
    
    args = parser.parse_args()
    main(args)