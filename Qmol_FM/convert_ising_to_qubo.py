# File: convert_ising_to_qubo.py

import numpy as np
import argparse
import logging
import os
import h5py

# Configure the main logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def convert_and_save(
    ising_solutions_csv_path: str,
    output_h5_path: str,
    x_name: str = "latent_codes"
):
    """
    Reads a CSV file of Ising solutions, processes them by handling an auxiliary spin,
    converts them to QUBO format, and saves both formats to an HDF5 file.
    """
    # --- Step 1: Validate and Load Input File ---
    if not os.path.exists(ising_solutions_csv_path):
        log.error(f"Error: Input CSV file not found at '{ising_solutions_csv_path}'")
        return

    log.info(f"Loading Ising solutions from '{ising_solutions_csv_path}'...")
    try:
        # Assumes the first row is a header (e.g., '0', '1', '2', ...)
        # and the delimiter is a comma.
        raw_ising_solutions = np.loadtxt(ising_solutions_csv_path, delimiter=",", skiprows=1)
        if raw_ising_solutions.ndim == 1: # Handle case with only one solution
            raw_ising_solutions = raw_ising_solutions.reshape(1, -1)
        n_solutions, n_variables_full = raw_ising_solutions.shape
        log.info(f"Loaded {n_solutions} solutions, each with {n_variables_full} variables (including auxiliary spin).")
    except Exception as e:
        log.error(f"Failed to load or parse CSV file: {e}")
        log.error("Please ensure the CSV has a header row and uses commas as delimiters.")
        return

    # --- Step 2: Process and Convert Each Solution ---
    log.info("Processing solutions: normalizing auxiliary spin and converting to QUBO...")
    
    processed_ising_list = []
    qubo_list = []

    for raw_solution in raw_ising_solutions:
        # This logic is taken directly from your original script.
        # It assumes the last variable is an auxiliary spin.
        
        # Flip the entire solution if the auxiliary spin is -1 to normalize it to +1
        s_full = raw_solution.copy()
        if s_full.shape[0] > 0 and s_full[-1] == -1:
            s_full = -s_full

        # The processed Ising solution (full length) is stored
        processed_ising_list.append(s_full)

        # Remove the auxiliary spin for the final sample
        if s_full.shape[0] > 1:
            ising_sample = s_full[:-1]
        else:
            ising_sample = s_full # Should not happen if n_variables > 1
            
        # Convert the core sample from Ising format (-1, +1) to QUBO format (0, 1)
        qubo_sample = ((ising_sample + 1) // 2).astype(np.int8)
        qubo_list.append(qubo_sample)

    log.info(f"Conversion complete. QUBO solutions will have {qubo_list[0].shape[0]} variables.")

    # --- Step 3: Save Solutions to HDF5 ---
    log.info(f"Saving processed Ising and QUBO solutions to HDF5 file: '{output_h5_path}'")
    try:
        # Convert lists of arrays to 2D NumPy matrices
        processed_ising_np = np.vstack(processed_ising_list).astype(np.int8)
        qubo_np = np.vstack(qubo_list).astype(np.int8)

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_h5_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with h5py.File(output_h5_path, 'w') as hf:
            # Save the full-length, processed Ising solutions
            hf.create_dataset("ising_codes", data=processed_ising_np, compression="gzip")
            
            # Save the QUBO solutions (without auxiliary spin)
            hf.create_dataset(x_name, data=qubo_np, compression="gzip")
            
            # Note: 'energies' dataset is not created as the original Ising matrix is not available.

        log.info("HDF5 file saved successfully!")
        log.info(f" -> Dataset 'ising_codes' shape: {processed_ising_np.shape}")
        log.info(f" -> Dataset '{x_name}' shape: {qubo_np.shape}")

    except Exception as e:
        log.error(f"Failed to save HDF5 file: {e}", exc_info=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert a CSV of Ising solutions to a QUBO-formatted HDF5 file.")
    parser.add_argument(
        "--input_csv", 
        type=str, 
        required=True, 
        help="Path to the input CSV file containing Ising solutions. Assumes a header row."
    )
    parser.add_argument(
        "--output_h5", 
        type=str, 
        required=True, 
        help="Path to the output HDF5 file for storing solution vectors."
    )
    parser.add_argument(
        "--x_name", 
        type=str, 
        default="latent_codes", 
        help="Dataset name for the QUBO solutions in the output HDF5 file."
    )

    args = parser.parse_args()
    convert_and_save(
        ising_solutions_csv_path=args.input_csv,
        output_h5_path=args.output_h5,
        x_name=args.x_name
    )
    log.info("\nConversion task completed!")