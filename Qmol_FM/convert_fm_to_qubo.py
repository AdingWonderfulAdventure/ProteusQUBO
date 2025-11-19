
import torch
import h5py
import numpy as np
import argparse
import logging
import os
from pathlib import Path

# Import the actual model class
from train_Qmol_FM import FMSurrogate
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def convert_fm_to_qubo_csv(model_path: str, output_csv_path: str):
    """
    Load a trained FM model checkpoint, convert its parameters to an upper triangular QUBO matrix,
    and save as a CSV file.

    Args:
        model_path (str): Path to PyTorch Lightning checkpoint (.ckpt).
        output_csv_path (str): Output CSV file path.
    """
    try:
        # --- Step 1: Load model and extract parameters ---
        log.info(f"Loading FM model from '{model_path}'...")
        # Set strict mode to False to handle minor inconsistencies with saved model
        model = FMSurrogate.load_from_checkpoint(model_path, map_location='cpu', strict=False)
        model.eval()

        # Extract weights and latent vectors
        V = model.model.V.detach().cpu().numpy()
        w = model.model.lin.weight.detach().cpu().numpy().squeeze()
        bias = model.model.lin.bias.item() if model.model.lin.bias is not None else 0.0

        n_features = V.shape[0]
        log.info(f"Model loaded successfully. QUBO dimension will be {n_features}x{n_features}.")

        # --- Step 2: Build QUBO matrix (upper triangular form) ---
        log.info("Converting FM parameters to QUBO matrix...")

        # Create an n x n zero matrix
        qubo_matrix = np.zeros((n_features, n_features), dtype=np.float64)

        # Compute off-diagonal terms (quadratic terms)
        # V @ V.T computes all dot products vi · vj
        interaction_matrix = V @ V.T

        # Fill upper triangular part
        for i in range(n_features):
            for j in range(i + 1, n_features):
                qubo_matrix[i, j] = -interaction_matrix[i, j]

        # Compute diagonal terms (linear terms)
        # Includes original linear weights and corrections from expanded cross terms
        diag_terms = -w
        for i in range(n_features):
            qubo_matrix[i, i] = diag_terms[i]

        log.info("QUBO matrix construction complete.")

        # --- Step 3: Save as CSV file ---
        output_path = Path(output_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists

        log.info(f"Saving upper triangular QUBO matrix to '{output_path}'...")
        # Use np.savetxt for precise format control
        np.savetxt(output_path, qubo_matrix, delimiter=",", fmt='%.8f')

        log.info("Conversion and save task completed successfully!")

    except FileNotFoundError:
        log.error(f"Error: Model file not found at '{model_path}'")
    except Exception as e:
        log.error(f"Unknown error occurred: {e}", exc_info=True)

    import neal
    import pandas as pd

    # --- a) Solve QUBO ---
    sampler = neal.SimulatedAnnealingSampler()
    num_reads = 1000  # Get more samples to observe distribution

    # Convert QUBO matrix to dictionary format, the standard format for DWave Ocean tools
    qubo_dict = {}
    for i in range(n_features):
        for j in range(i, n_features):
            if qubo_matrix[i, j] != 0:
                qubo_dict[(i, j)] = qubo_matrix[i, j]

    log.info(f"Running simulated annealing with {num_reads} samples...")
    response = sampler.sample_qubo(qubo_dict, num_reads=num_reads)

    log.info("Solving complete.")

    # --- b) Analyze and verify results ---
    log.info("--- Starting verification of all solutions ---")

    results = []
    num_samples_to_verify = len(response.record) # Get total number of returned solutions
    log.info(f"Solver returned {num_samples_to_verify} samples (including duplicates), verifying each...")

    with torch.no_grad(): # Place torch.no_grad() outside loop for efficiency
        for i in range(num_samples_to_verify):
            # Get information for each sample from response.record
            sample = response.record[i].sample
            energy = response.record[i].energy
            num_occurrences = response.record[i].num_occurrences

            # Convert sample dictionary to numpy vector
            solution_vector = np.array([sample[j] for j in range(n_features)])

            # Calculate score using original FM model
            solution_torch = torch.from_numpy(solution_vector).float().unsqueeze(0)
            fm_score = model(solution_torch).item()

            # Reconstruct score from energy
            reconstructed_score = -energy + bias

            # Check if relationship holds
            is_match = np.isclose(fm_score, reconstructed_score, atol=1e-5)

            results.append({
                'sample_index': i,
                'qubo_energy': energy,
                'fm_score': fm_score,
                'reconstructed_fm_score': reconstructed_score,
                'is_match': is_match,
                'num_occurrences': num_occurrences
            })

    # --- c) Results summary and report ---
    results_df = pd.DataFrame(results)

    # Count mismatches
    mismatched_count = results_df['is_match'].eq(False).sum()

    log.info("\n" + "="*60)
    log.info("Solution Verification Report")
    log.info("="*60)
    log.info(f"Total samples verified: {num_samples_to_verify}")
    log.info(f"Energy-score relationship mismatches: {mismatched_count}")

    if mismatched_count == 0:
        log.info("All solution energy-score relationships (y ≈ -E + b) verified successfully!")
    else:
        log.error(f"Warning: Found {mismatched_count} mismatched solutions!")
        log.error("This may indicate floating-point precision issues or logic errors in FM model implementation or QUBO conversion.")
        # Print first 5 mismatched samples for debugging
        log.info("--- Details of first 5 mismatched samples ---")
        pd.set_option('display.width', 120)
        print(results_df[results_df['is_match'] == False].head(5))

    # Display verification results for top 5 lowest energy solutions
    log.info("\n--- Verification details for top 5 lowest energy solutions ---")
    # DWave's response.record is sorted by energy by default
    print(results_df.head(5).to_string(index=False))
    log.info("="*60)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert PyTorch Lightning FM model to QUBO matrix CSV file.")
    parser.add_argument("--model_path", type=str, help="Path to input PyTorch Lightning model checkpoint file (.ckpt).")
    parser.add_argument("--output_csv", type=str, help="Path to output upper triangular QUBO matrix CSV file.")

    args = parser.parse_args()

    convert_fm_to_qubo_csv(args.model_path, args.output_csv)