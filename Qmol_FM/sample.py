# File: solve_ising_model.py

import numpy as np
import argparse
import logging
import os
import time
import h5py

# --- Import Kaiwu SDK ---
try:
    import kaiwu as kw
    import kaiwu.classical as kaiwu_classical
except ImportError:
    print("Error: kaiwusdk is not installed. Please install it according to the official documentation.")
    exit()

# Configure the main logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def setup_file_logger(log_file_path: str):
    """Configures a dedicated file logger for solution details."""
    file_logger = logging.getLogger('solutions_logger')
    file_logger.setLevel(logging.INFO)
    file_logger.propagate = False
    if file_logger.hasHandlers():
        file_logger.handlers.clear()
    handler = logging.FileHandler(log_file_path, mode='w')
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    file_logger.addHandler(handler)
    return file_logger

def solve_ising_and_save(
    ising_csv_path: str,
    output_h5_path: str,
    output_log_path: str,
    user_id: str,
    sdk_code: str,
    num_solutions: int,
    x_name: str = "latent_codes"
):
    """
    Reads an Ising matrix, solves it using the Kaiwu SDK,
    saves the solutions in both QUBO (0/1) and raw Ising (-1/+1) formats,
    and logs detailed solution information.
    """
    if not os.path.exists(ising_csv_path):
        log.error(f"Error: Ising file not found at '{ising_csv_path}'")
        return

    # --- Step 1: Initialize SDK ---
    log.info("Initializing Kaiwu SDK...")
    kw.license.init(user_id=user_id, sdk_code=sdk_code)

    # --- Step 2: Load Ising Matrix ---
    log.info(f"Loading Ising matrix from '{ising_csv_path}'...")
    ising_matrix = np.loadtxt(ising_csv_path, delimiter=",")
    n_variables = ising_matrix.shape[0]
    log.info(f"Ising matrix loaded successfully. Problem dimension: {n_variables}x{n_variables}.")

    # --- Step 3: Configure and Run the Solver ---
    log.info("Configuring Simulated Annealing optimizer for Ising model...")
    sampler = kaiwu_classical.SimulatedAnnealingOptimizer(
        initial_temperature=1000.0, alpha=0.9999, cutoff_temperature=0.0001,
        iterations_per_t=5000, size_limit=num_solutions,
    )# Using Simulated Annealing for Ising problems 100 0.999 0.001 300 

    log.info(f"Starting to solve the Ising model... Will return the top {num_solutions} optimal solutions.")
    start_time = time.time()
    raw_solutions = sampler.solve(ising_matrix)  # Input is the raw Ising matrix
    elapsed_time = time.time() - start_time
    log.info(f"Solving completed in {elapsed_time:.2f} seconds. Found {len(raw_solutions)} solutions.")

    if len(raw_solutions) == 0:
        log.warning("The solver did not return any solutions. Task terminated.")
        return

    # --- Step 4: Parse and Prepare Solutions for Saving ---
    log.info("Parsing solutions and generating HDF5 and log files...")
    solutions_log = setup_file_logger(output_log_path)
    solutions_log.info("=" * 50)
    solutions_log.info("Ising Solver Results")
    solutions_log.info("=" * 50)

    solutions_data = []
    feature_names = [f"feature_{i}" for i in range(n_variables - 1)]

    for orig_idx, raw_solution in enumerate(raw_solutions):
        # raw_solution is in Ising format (-1, +1), the last variable is assumed to be an auxiliary spin.
        
        # Calculate energy in the Ising domain
        energy = -raw_solution @ ising_matrix @ raw_solution

        # Flip the solution if necessary to ensure the auxiliary spin is +1
        s_full = raw_solution.copy()
        if s_full.shape[0] > 0 and s_full[-1] == -1:
            s_full = -s_full

        # Remove the auxiliary spin for the final sample
        if s_full.shape[0] > 1:
            ising_sample = s_full[:-1]
        else:
            ising_sample = s_full
            
        # Convert to QUBO format (0/1)
        qubo_sample = ((ising_sample + 1) // 2).astype(np.int8)

        solutions_data.append((orig_idx, energy, s_full, qubo_sample))

    # --- Step 5: Sort Solutions by Energy ---
    sorted_solutions = sorted(solutions_data, key=lambda x: x[1])
    qubo_matrix_list = []
    ising_matrix_list = []
    energy_list = []

    for rank, (orig_idx, energy, processed_ising_sol, qubo_sample) in enumerate(sorted_solutions):
        qubo_matrix_list.append(qubo_sample)
        ising_matrix_list.append(processed_ising_sol)
        energy_list.append(energy)

        active_features = [feature_names[j] for j, val in enumerate(qubo_sample) if val == 1]

        solutions_log.info(
            f"\n--- Rank: {rank + 1} | Original Index: {orig_idx} | Energy (lower is better): {energy:.4f} ---"
        )
        solutions_log.info(f"  Processed Ising Solution: {processed_ising_sol.tolist()}")
        solutions_log.info(f"  QUBO Format:              {qubo_sample.tolist()}")
        solutions_log.info(f"  Active Features ({len(active_features)}): {active_features}")

    log.info(f"Detailed solution information has been written to log file: '{output_log_path}'")

    # --- Step 6: Save Solutions to HDF5 ---
    log.info(f"Saving Ising/QUBO solution matrices and energies to HDF5 file: '{output_h5_path}'")
    try:
        qubo_matrix_np = np.vstack(qubo_matrix_list).astype(np.int8)
        ising_matrix_np = np.vstack(ising_matrix_list).astype(np.int8)
        energy_array_np = np.array(energy_list, dtype=np.float32)

        output_dir = os.path.dirname(output_h5_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with h5py.File(output_h5_path, 'w') as hf:
            hf.create_dataset("ising_codes", data=ising_matrix_np, compression="gzip")
            hf.create_dataset(x_name, data=qubo_matrix_np, compression="gzip")
            hf.create_dataset("energies", data=energy_array_np, compression="gzip")

        log.info("HDF5 file saved successfully!")
    except Exception as e:
        log.error(f"Failed to save HDF5 file: {e}", exc_info=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Solve an Ising model using the Kaiwu SDK and output solutions in QUBO and Ising formats.")
    parser.add_argument("--ising_csv", type=str, required=True, help="Path to the input Ising matrix CSV file.")
    parser.add_argument("--output_h5", type=str, required=True, help="Path to the output HDF5 file for storing solution vectors.")
    parser.add_argument("--output_log", type=str, required=True, help="Path to the output log file for detailed solution information.")
    parser.add_argument("--user_id", type=str, required=True, help="Kaiwu SDK User ID.")
    parser.add_argument("--sdk_code", type=str, required=True, help="Kaiwu SDK Authorization Code.")
    parser.add_argument("--num_solutions", "-n", type=int, default=500, help="Number of optimal solutions to find and save.")
    parser.add_argument("--x_name", type=str, default="latent_codes", help="Dataset name for QUBO solutions in the output HDF5 file.")

    args = parser.parse_args()
    solve_ising_and_save(
        ising_csv_path=args.ising_csv,
        output_h5_path=args.output_h5,
        output_log_path=args.output_log,
        user_id=args.user_id,
        sdk_code=args.sdk_code,
        num_solutions=args.num_solutions,
        x_name=args.x_name
    )
    log.info("\nAll tasks completed!")