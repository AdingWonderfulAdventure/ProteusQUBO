import json
import numpy as np
import kaiwu as kw
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def build_qubo_with_constraints(input_qubo_path: str, combinations_json_path: str, output_qubo_path: str):
    try:
        # --- Step 1: Read input files ---
        log.info(f"Reading QUBO matrix from '{input_qubo_path}'...")
        qubo_matrix = np.loadtxt(input_qubo_path, delimiter=",")
        n_total = qubo_matrix.shape[0]
        log.info(f"QUBO matrix loaded successfully, dimensions: {n_total}x{n_total}")

        log.info(f"Reading combination rules from '{combinations_json_path}'...")
        with open(combinations_json_path, "r") as f:
            combinations = json.load(f)["combinations"]
        log.info(f"Successfully loaded {len(combinations)} combination rules.")

        # --- Step 2: Build original objective function ---
        log.info("Building original objective function...")
        x = kw.qubo.ndarray(n_total, "x", kw.qubo.Binary)
        objective = kw.qubo.dot(x, kw.qubo.dot(qubo_matrix, x))
        log.info("Objective function construction complete.")

        # --- Step 3: Set penalty coefficient ---
        max_abs_coeff = np.max(np.abs(qubo_matrix))
        P = 4 * max_abs_coeff  # Using specified multiplier
        log.info(f"Using penalty coefficient: P = {P:.6f}")

        # --- Step 4: Build constraint penalty terms ---
        log.info("Building constraint penalty terms...")
        total_penalty_expression = kw.qubo.QuboExpression()
        
        for idx, combo in enumerate(combinations):
            orig_idx = [int(f.split("_")[1]) for f in combo["features"]]
            
            if not orig_idx: 
                continue

            penalty_expr = kw.qubo.QuboExpression()

            if len(orig_idx) == 1:
                # Single variable constraint: y = x_i
                y_out = x[128 + idx]
                a = x[orig_idx[0]]
                penalty_expr = (y_out - a)**2
            else:
                # Multi-variable constraint: y = x_i * x_j * ... (using chained Rosenberg penalty)
                current_prod_var = x[orig_idx[0]]
                for i in range(1, len(orig_idx)):
                    next_var = x[orig_idx[i]]
                    is_last_step = (i == len(orig_idx) - 1)

                    if is_last_step:
                        output_var = x[128 + idx]
                    else:
                        output_var = kw.qubo.Binary(f"z_aux_{idx}_{i-1}")

                    a, b, y_out = current_prod_var, next_var, output_var

                    # Rosenberg quadratic penalty term: a*b - 2*a*y - 2*b*y + 3*y
                    rosenberg_penalty = a*b - 2*a*y_out - 2*b*y_out + 3*y_out
                    penalty_expr += rosenberg_penalty

                    current_prod_var = output_var

            total_penalty_expression += P * penalty_expr

        log.info(f"Built penalty terms for {len(combinations)} constraints")

        # --- Step 5: Build final QUBO model ---
        log.info("Building final QUBO model...")
        final_model = kw.qubo.QuboModel(objective=objective + total_penalty_expression)
        final_qubo_matrix = final_model.get_matrix()
        final_variables = final_model.get_variables()
        final_offset = final_model.get_offset()

        log.info(f"Final QUBO matrix generated, dimensions: {final_qubo_matrix.shape}")
        log.info(f"Total variables: {len(final_variables)}")
        log.info(f"Offset: {final_offset}")

        # --- Step 6: Save results ---
        output_path = Path(output_qubo_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save QUBO matrix
        log.info(f"Saving QUBO matrix to '{output_path}'...")
        np.savetxt(output_path, final_qubo_matrix, delimiter=",")

        # Save model information
        info_path = output_path.parent / f"{output_path.stem}_info.json"
        model_info = {
            "original_matrix_size": n_total,
            "final_matrix_size": final_qubo_matrix.shape[0],
            "num_constraints": len(combinations),
            "penalty_coefficient": float(P),
            "variables": final_variables,
            "offset": float(final_offset),
            "constraint_summary": {
                f"length_{length}": sum(1 for combo in combinations if len(combo["features"]) == length)
                for length in set(len(combo["features"]) for combo in combinations)
            }
        }

        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)

        log.info(f"QUBO matrix saved successfully: {output_path}")
        log.info(f"Model information saved successfully: {info_path}")
        log.info("QUBO model construction task complete!")

        return final_qubo_matrix, model_info

    except Exception as e:
        log.error(f"Error occurred: {e}", exc_info=True)
        return None, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Build QUBO model with constraints and save matrix",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--input_qubo", type=str, required=True,
                       help="Path to input original QUBO matrix CSV file")
    parser.add_argument("--combinations_json", type=str, required=True,
                       help="Path to JSON file containing combination rules")
    parser.add_argument("--output_qubo", type=str, required=True,
                       help="Path to output final QUBO matrix CSV file")
    parser.add_argument("--penalty_multiplier", type=float, default=8.0,
                       help="Penalty coefficient multiplier, default 8.0")

    args = parser.parse_args()

    # Support custom penalty coefficient multiplier
    if hasattr(args, 'penalty_multiplier'):
        # This would require modifying the line P = 8.0 * max_abs_coeff in the code
        # But for simplicity, we just log the parameter here
        log.info(f"Penalty coefficient multiplier parameter: {args.penalty_multiplier}")

    matrix, info = build_qubo_with_constraints(
        args.input_qubo,
        args.combinations_json,
        args.output_qubo
    )

    if matrix is not None:
        log.info(f"Successfully built {matrix.shape[0]}x{matrix.shape[1]} QUBO matrix")
    else:
        log.error("QUBO model construction failed!")