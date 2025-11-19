"""
Feature-guided optimizer v2 (QUBO native version) - v1.1

Core strategies:
1. Single feature optimization: Flip top-K most important features one by one
2. Pair configuration optimization: Apply optimal value configurations for feature pairs
3. Diversified generation: Generate multiple optimized variants for each input molecule

Input data:
- single_features_detailed.json: Single features and their optimal values
- pair_value_configurations.json: 4 configurations for feature pairs and their impacts
==============================================================================
"""

import argparse
import h5py
import numpy as np
import logging
import os
import json
from tqdm import tqdm
from typing import List, Dict, Tuple
from collections import defaultdict
from joblib import Parallel, delayed
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


class MultiVariantOptimizer:
    """Multi-variant optimizer: Generate multiple optimized variants from a single solution (QUBO native)"""
    
    def __init__(self, 
                 qubo_matrix: np.ndarray,
                 single_features_json: str,
                 pair_configs_json: str):
        """
        Initialize optimizer

        Args:
            qubo_matrix: QUBO matrix (Q)
            single_features_json: Single feature data
            pair_configs_json: Feature pair configuration data
        """
        self.qubo_matrix = qubo_matrix
        self.n_qubo_vars = qubo_matrix.shape[0]

        # Load feature data
        self.single_features = self._load_single_features(single_features_json)
        self.pair_configs = self._load_pair_configs(pair_configs_json)

        log.info(f"Optimizer initialized:")
        log.info(f"  - QUBO dimension: {self.n_qubo_vars}")
        log.info(f"  - Single features: {len(self.single_features)}")
        log.info(f"  - Feature pairs: {len(self.pair_configs)}")
    
    def _load_single_features(self, json_path: str) -> List[Dict]:
        """Load single feature data"""
        if not json_path or not os.path.exists(json_path):
            log.warning(f"Single feature file not found: {json_path}")
            return []

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            features = data.get('features', [])
            log.info(f"Successfully loaded {len(features)} single features")
            return features
        except Exception as e:
            log.error(f"Failed to load single features: {e}")
            return []
    
    def _load_pair_configs(self, json_path: str) -> List[Dict]:
        """Load feature pair configuration data"""
        if not json_path or not os.path.exists(json_path):
            log.warning(f"Feature pair config file not found: {json_path}")
            return []

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            pairs = data.get('pairs', [])
            log.info(f"Successfully loaded {len(pairs)} feature pair configs")
            return pairs
        except Exception as e:
            log.error(f"Failed to load feature pair configs: {e}")
            return []
    
    def _parse_feature_name(self, feature_name: str) -> int:
        """Extract index from feature name"""
        import re
        match = re.search(r'\d+$', feature_name)
        if match:
            return int(match.group())
        return -1
    
    def _calculate_energy(self, qubo_solution: np.ndarray) -> float:
        """Calculate QUBO energy E = x^T Q x"""
        return float(qubo_solution @ self.qubo_matrix @ qubo_solution)
    
    def generate_single_feature_variants(self,
                                        qubo_solution: np.ndarray,
                                        top_k: int = 10,
                                        min_improvement: float = 1e-10) -> List[Tuple[np.ndarray, float, str]]:
        """
        Generate single feature variants: Flip most important single features one by one

        Returns:
            List[(optimized solution, energy, description)]
        """
        if not self.single_features:
            return []
        
        original_energy = self._calculate_energy(qubo_solution)
        variants = []
        
        for feat_data in self.single_features[:top_k]:
            feat_name = feat_data['feature']
            feat_idx = self._parse_feature_name(feat_name)
            
            if feat_idx < 0 or feat_idx >= self.n_qubo_vars:
                continue
            
            contrib_0 = feat_data.get('value_0_avg_contribution', 0)
            contrib_1 = feat_data.get('value_1_avg_contribution', 0)
            
            target_value = 1 if contrib_1 < contrib_0 else 0
            current_contrib = contrib_1 if qubo_solution[feat_idx] == 1 else contrib_0
            expected_improvement = current_contrib - min(contrib_0, contrib_1)

            if qubo_solution[feat_idx] == target_value:
                continue
            
            variant = qubo_solution.copy()
            variant[feat_idx] = target_value
            variant_energy = self._calculate_energy(variant)
            
            energy_improvement = original_energy - variant_energy
            
            if energy_improvement > min_improvement:
                description = (f"SingleFlip_x{feat_idx}_{qubo_solution[feat_idx]}to{target_value}_"
                              f"ΔE{energy_improvement:.6f}_ExpΔ{expected_improvement:.6f}")
                variants.append((variant, variant_energy, description))
        
        return variants
    
    def generate_pair_config_variants(self,
                                     qubo_solution: np.ndarray,
                                     top_k: int = 20,
                                     min_improvement: float = 1e-10) -> List[Tuple[np.ndarray, float, str]]:
        """
        Generate configuration variants: Apply optimal configurations for feature pairs

        Returns:
            List[(optimized solution, energy, description)]
        """
        if not self.pair_configs:
            return []
        
        original_energy = self._calculate_energy(qubo_solution)
        variants = []
        
        for pair_data in self.pair_configs[:top_k]:
            feat1_name = pair_data['feature_1']
            feat2_name = pair_data['feature_2']
            
            feat1_idx = self._parse_feature_name(feat1_name)
            feat2_idx = self._parse_feature_name(feat2_name)
            
            if feat1_idx < 0 or feat1_idx >= self.n_qubo_vars or \
               feat2_idx < 0 or feat2_idx >= self.n_qubo_vars:
                continue
            
            configs = pair_data['configurations']
            best_config_name = min(configs.keys(), 
                                  key=lambda k: configs[k].get('avg_contribution', 0))
            best_config_data = configs[best_config_name]
            
            v1, v2 = eval(best_config_name)
            
            if qubo_solution[feat1_idx] == v1 and qubo_solution[feat2_idx] == v2:
                continue
            
            variant = qubo_solution.copy()
            variant[feat1_idx] = v1
            variant[feat2_idx] = v2
            variant_energy = self._calculate_energy(variant)
            
            energy_improvement = original_energy - variant_energy
            
            if energy_improvement > min_improvement:
                current_config_key = f"({qubo_solution[feat1_idx]}, {qubo_solution[feat2_idx]})"
                current_contrib = configs.get(current_config_key, {}).get('avg_contribution', 0)
                expected_improvement = current_contrib - best_config_data['avg_contribution']
                
                description = (f"PairConfig_x{feat1_idx}x{feat2_idx}_"
                              f"{qubo_solution[feat1_idx]}{qubo_solution[feat2_idx]}to{v1}{v2}_"
                              f"ΔE{energy_improvement:.6f}_"
                              f"ExpΔ{expected_improvement:.6f}")
                variants.append((variant, variant_energy, description))
        
        return variants
    
    def generate_cumulative_variants(self,
                                qubo_solution: np.ndarray,
                                top_k_singles: int = 5,
                                top_k_pairs: int = 5,
                                min_improvement: float = 1e-10) -> List[Tuple[np.ndarray, float, str]]:
        """
        [v1.2 LOGIC] Generate cumulative variants: Use adaptive greedy search strategy.
        At each step, find the best modification for the current solution instead of pre-selecting candidates.
        """
        original_energy = self._calculate_energy(qubo_solution)
        variants = []

        # --- Path 1: Greedy cumulative application of single feature modifications ---
        current_solution_s = qubo_solution.copy()
        applied_modifications_s = []
        for i in range(top_k_singles):
            best_mod_s = None
            best_mod_energy = self._calculate_energy(current_solution_s)

            # Iterate through all possible single bit flips, find the best one
            for feat_data in self.single_features:
                feat_idx = self._parse_feature_name(feat_data['feature'])
                if feat_idx < 0 or feat_idx >= self.n_qubo_vars:
                    continue

                # Try flipping
                temp_solution = current_solution_s.copy()
                temp_solution[feat_idx] = 1 - temp_solution[feat_idx] # Flip the bit
                temp_energy = self._calculate_energy(temp_solution)

                if temp_energy < best_mod_energy:
                    best_mod_energy = temp_energy
                    best_mod_s = (temp_solution.copy(), feat_idx, 1 - current_solution_s[feat_idx])

            # If improvement found, apply it and save variant
            if best_mod_s is not None:
                current_solution_s, feat_idx, target_val = best_mod_s
                applied_modifications_s.append(f"x{feat_idx}→{target_val}")

                improvement = original_energy - best_mod_energy
                if improvement > min_improvement:
                    description = f"GreedyCumulative_Single_{i+1}mods_{'_'.join(applied_modifications_s)}_ΔE{improvement:.6f}"
                    variants.append((current_solution_s.copy(), best_mod_energy, description))
            else:
                # If no improvement found, terminate this path early
                break

        # --- Path 2: Greedy cumulative application of config modifications (similar logic) ---
        current_solution_p = qubo_solution.copy()
        applied_modifications_p = []
        # Track modified variables to avoid duplicate modifications in same step
        modified_indices_in_run = set()

        for i in range(top_k_pairs):
            best_mod_p = None
            best_mod_energy = self._calculate_energy(current_solution_p)

            # Iterate through all feature pairs, find best config to apply
            for pair_data in self.pair_configs:
                feat1_idx = self._parse_feature_name(pair_data['feature_1'])
                feat2_idx = self._parse_feature_name(pair_data['feature_2'])

                # Skip already modified variables
                if feat1_idx in modified_indices_in_run or feat2_idx in modified_indices_in_run:
                    continue
                if feat1_idx < 0 or feat2_idx < 0: continue

                configs = pair_data['configurations']
                best_config_name = min(configs.keys(), key=lambda k: configs[k].get('avg_contribution', 0))
                v1, v2 = eval(best_config_name)

                if current_solution_p[feat1_idx] == v1 and current_solution_p[feat2_idx] == v2:
                    continue

                temp_solution = current_solution_p.copy()
                temp_solution[feat1_idx] = v1
                temp_solution[feat2_idx] = v2
                temp_energy = self._calculate_energy(temp_solution)

                if temp_energy < best_mod_energy:
                    best_mod_energy = temp_energy
                    best_mod_p = (temp_solution.copy(), feat1_idx, feat2_idx, v1, v2)

            if best_mod_p is not None:
                current_solution_p, f1, f2, v1, v2 = best_mod_p
                modified_indices_in_run.add(f1)
                modified_indices_in_run.add(f2)
                applied_modifications_p.append(f"x{f1}x{f2}→{v1}{v2}")

                improvement = original_energy - best_mod_energy
                if improvement > min_improvement:
                    description = f"GreedyCumulative_Pair_{i+1}mods_{'_'.join(applied_modifications_p)}_ΔE{improvement:.6f}"
                    variants.append((current_solution_p.copy(), best_mod_energy, description))
            else:
                break
                
        return variants

    def optimize_with_variants(self,
                              qubo_solution: np.ndarray,
                              single_top_k: int = 10,
                              pair_top_k: int = 20,
                              cumulative_singles: int = 5,
                              cumulative_pairs: int = 5,
                              max_variants_per_solution: int = 50) -> List[Tuple[np.ndarray, float, str]]:
        """
        Comprehensively generate multiple optimized variants

        Returns:
            List[(optimized solution, energy, description)], sorted by energy improvement
        """
        all_variants = []

        # 1. Single feature variants
        single_variants = self.generate_single_feature_variants(qubo_solution, top_k=single_top_k)
        all_variants.extend(single_variants)

        # 2. Configuration variants
        pair_variants = self.generate_pair_config_variants(qubo_solution, top_k=pair_top_k)
        all_variants.extend(pair_variants)

        # 3. Cumulative variants
        cumulative_variants = self.generate_cumulative_variants(qubo_solution,
                                                               top_k_singles=cumulative_singles,
                                                               top_k_pairs=cumulative_pairs)
        all_variants.extend(cumulative_variants)

        # Deduplicate (based on solution content)
        unique_variants = []
        seen_solutions = set()

        for variant, energy, desc in all_variants:
            solution_hash = variant.tobytes()
            if solution_hash not in seen_solutions:
                seen_solutions.add(solution_hash)
                unique_variants.append((variant, energy, desc))

        # Sort by energy (lower is better)
        unique_variants.sort(key=lambda x: x[1])

        # Limit quantity
        return unique_variants[:max_variants_per_solution]

# ======================= Worker function (with Hamming distance) =======================
def process_single_solution(
    optimizer: MultiVariantOptimizer,
    original_sol: np.ndarray,
    original_label: str,
    parent_idx: int,
    config: dict
) -> Tuple[List, List, List, List, List, Dict]:
    """
    Process a single initial solution, generate all variants and return results.
    This is the core function executed in each parallel process.
    """
    solutions, energies, labels, parent_indices, hamming_distances = [], [], [], [], []
    stats_single = defaultdict(int)

    original_energy = optimizer._calculate_energy(original_sol)

    variants = optimizer.optimize_with_variants(
        original_sol,
        config['single_top_k'], config['pair_top_k'],
        config['cumulative_singles'], config['cumulative_pairs'],
        config['max_variants_per_solution']
    )

    for variant_idx, (variant_sol, variant_energy, description) in enumerate(variants):
        solutions.append(variant_sol)
        energies.append(variant_energy)
        labels.append(f"{original_label}_{variant_idx + 1}")
        parent_indices.append(parent_idx)

        # Calculate and store Hamming distance
        distance = np.sum(original_sol != variant_sol)
        hamming_distances.append(distance)

        # Collect statistics
        stats_single['total_variants'] += 1
        if variant_energy < original_energy:
            stats_single['improved_variants'] += 1
        if 'SingleFlip' in description: stats_single['single_feature_variants'] += 1
        elif 'PairConfig' in description: stats_single['pair_config_variants'] += 1
        elif 'GreedySingle' in description: stats_single['cumulative_single_variants'] += 1
        elif 'GreedyPair' in description: stats_single['cumulative_pair_variants'] += 1

    return solutions, energies, labels, parent_indices, hamming_distances, stats_single

def optimize_solutions_batch(
    qubo_csv_path: str,
    input_h5_path: str,
    output_h5_path: str,
    single_features_json: str,
    pair_configs_json: str,
    n_jobs: int,
    single_top_k: int,
    pair_top_k: int,
    cumulative_singles: int,
    cumulative_pairs: int,
    max_variants_per_solution: int,
    latent_dataset_name: str
):
    log.info("=" * 80)
    log.info(f"Multi-variant feature-guided optimization workflow (v1.4 - parallel acceleration, n_jobs={n_jobs})")
    log.info("=" * 80)

    try:
        log.info(f"Loading QUBO matrix: {qubo_csv_path}")
        qubo_matrix = np.loadtxt(qubo_csv_path, delimiter=",")
    except Exception as e:
        log.error(f"Failed to load QUBO matrix: {e}")
        return

    try:
        log.info(f"Loading initial solutions: {input_h5_path}")
        with h5py.File(input_h5_path, 'r') as hf:
            initial_solutions = hf[latent_dataset_name][:]
            if 'labels' in hf:
                raw_labels = hf['labels'][:]
                initial_labels = [label.decode('utf-8') if isinstance(label, bytes) else str(label) for label in raw_labels]
            else:
                initial_labels = [f"original_{i}" for i in range(len(initial_solutions))]
        num_initial = len(initial_solutions)
        log.info(f"Loaded {num_initial} initial solutions and labels")
    except Exception as e:
        log.error(f"Failed to load initial solutions: {e}", exc_info=True)
        return

    log.info("\nInitializing multi-variant optimizer (shared across all processes)...")
    optimizer = MultiVariantOptimizer(qubo_matrix, single_features_json, pair_configs_json)

    # Package config for worker function
    config = {
        'single_top_k': single_top_k, 'pair_top_k': pair_top_k,
        'cumulative_singles': cumulative_singles, 'cumulative_pairs': cumulative_pairs,
        'max_variants_per_solution': max_variants_per_solution
    }
    log.info(f"\nOptimization config: {config}")
    log.info("-" * 80)

    # Pre-compute original energies for statistics
    original_energies_list = [optimizer._calculate_energy(sol) for sol in initial_solutions]

    # ======================= Parallel processing core =======================
    log.info(f"Starting parallel variant generation for {num_initial} solutions...\n")
    
    tasks = [delayed(process_single_solution)(
                optimizer,
                initial_solutions[i],
                initial_labels[i],
                i,
                config
             ) for i in range(num_initial)]

    # Use n_jobs to control parallelism, -1 means use all CPU cores
    # backend="loky" is joblib's default option, more robust
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        tqdm(tasks, desc="Generating variants")
    )
    # =============================================================

    # ======================= Result aggregation =======================
    log.info("All parallel tasks completed, aggregating results...")
    all_solutions, all_energies, all_labels, all_parent_indices, all_hamming_distances = [], [], [], [], []
    stats = defaultdict(int)

    for sol_list, en_list, lbl_list, p_idx_list, dist_list, stats_single in results:
        all_solutions.extend(sol_list)
        all_energies.extend(en_list)
        all_labels.extend(lbl_list)
        all_parent_indices.extend(p_idx_list)
        all_hamming_distances.extend(dist_list)
        for key, value in stats_single.items():
            stats[key] += value
    # ==========================================================

    log.info("\n" + "=" * 80)
    log.info("Optimization statistics:")
    log.info(f"  Input solutions: {num_initial}")
    log.info(f"  Total output solutions: {len(all_solutions)} (variants only)")
    log.info(f"  Generated variants: {stats['total_variants']}")
    if stats['total_variants'] > 0:
      log.info(f"  Improved variants: {stats['improved_variants']} ({stats['improved_variants']/stats['total_variants']*100:.1f}%)")
    log.info(f"\nVariant type distribution:")
    log.info(f"  - Single feature variants: {stats['single_feature_variants']}")
    log.info(f"  - Config variants: {stats['pair_config_variants']}")
    log.info(f"  - Cumulative (single) variants: {stats['cumulative_single_variants']}")
    log.info(f"  - Cumulative (pair) variants: {stats['cumulative_pair_variants']}")
    if num_initial > 0:
      log.info(f"  Average variants per input: {stats['total_variants']/num_initial:.1f}")

    if original_energies_list and all_energies:
        original_energies = np.array(original_energies_list)
        variant_energies = np.array(all_energies)
        log.info(f"\nEnergy distribution:")
        log.info(f"  Original solution energy range: [{original_energies.min():.6f}, {original_energies.max():.6f}], avg: {original_energies.mean():.6f}")
        log.info(f"  Variant energy range: [{variant_energies.min():.6f}, {variant_energies.max():.6f}], avg: {variant_energies.mean():.6f}")
        log.info(f"  Best improvement: {original_energies.min() - variant_energies.min():.6f}")

    if all_hamming_distances:
        distances = np.array(all_hamming_distances)
        log.info(f"\nModified variables statistics (Hamming Distance):")
        log.info(f"  - Min modified bits: {distances.min()} / {optimizer.n_qubo_vars}")
        log.info(f"  - Max modified bits: {distances.max()} / {optimizer.n_qubo_vars}")
        log.info(f"  - Avg modified bits: {distances.mean():.2f} / {optimizer.n_qubo_vars}")

    # Save results
    try:
        log.info(f"\nSaving results to: {output_h5_path}")
        os.makedirs(os.path.dirname(output_h5_path) if os.path.dirname(output_h5_path) else '.', exist_ok=True)
        with h5py.File(output_h5_path, 'w') as hf:
            string_dt = h5py.string_dtype(encoding='utf-8')
            hf.create_dataset(latent_dataset_name, data=np.array(all_solutions, dtype=np.int8), compression="gzip")
            hf.create_dataset("energies", data=np.array(all_energies, dtype=np.float32), compression="gzip")
            hf.create_dataset("labels", data=np.array(all_labels, dtype=object), dtype=string_dt, compression="gzip")
            hf.create_dataset("parent_indices", data=np.array(all_parent_indices, dtype=np.int32), compression="gzip")
            hf.create_dataset("hamming_distances", data=np.array(all_hamming_distances, dtype=np.uint8), compression="gzip")
            hf.attrs['num_input_solutions'] = num_initial
            hf.attrs['num_total_solutions'] = len(all_solutions)
        log.info(f"Successfully saved {len(all_solutions)} solutions")
    except Exception as e:
        log.error(f"Save failed: {e}", exc_info=True)
        return
    log.info("=" * 80)
    log.info("Multi-variant optimization workflow completed!")
    log.info("=" * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Feature analysis-based multi-variant optimizer (QUBO native v1.4 - parallel acceleration)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Parameter definitions
    parser.add_argument('--qubo_csv', type=str, required=True, help='QUBO matrix CSV file')
    parser.add_argument('--input_h5', type=str, required=True, help='Input HDF5 file')
    parser.add_argument('--output_h5', type=str, required=True, help='Output HDF5 file')
    parser.add_argument('--single_features_json', type=str, required=True, help='Single feature JSON file')
    parser.add_argument('--pair_configs_json', type=str, required=True, help='Feature pair config JSON file')

    parser.add_argument('--single_top_k', type=int, default=10, help='Use top K most important single features')
    parser.add_argument('--pair_top_k', type=int, default=10, help='Use top K most important feature pair configs')
    parser.add_argument('--cumulative_singles', type=int, default=5, help='Number of single features in cumulative optimization')
    parser.add_argument('--cumulative_pairs', type=int, default=5, help='Number of feature pairs in cumulative optimization')
    parser.add_argument('--max_variants', type=int, default=20, help='Max variants generated per input solution')
    parser.add_argument('--dataset_name', type=str, default='latent_codes', help='HDF5 dataset name')

    # Parallel parameter
    parser.add_argument('--n_jobs', type=int, default=192, help='Number of CPU cores for parallelism (-1 for all cores)')

    args = parser.parse_args()

    for file_path in [args.qubo_csv, args.input_h5, args.single_features_json, args.pair_configs_json]:
        if not os.path.exists(file_path):
            log.error(f"File not found: {file_path}")
            exit(1)

    optimize_solutions_batch(
        qubo_csv_path=args.qubo_csv,
        input_h5_path=args.input_h5,
        output_h5_path=args.output_h5,
        single_features_json=args.single_features_json,
        pair_configs_json=args.pair_configs_json,
        n_jobs=args.n_jobs,
        single_top_k=args.single_top_k,
        pair_top_k=args.pair_top_k,
        cumulative_singles=args.cumulative_singles,
        cumulative_pairs=args.cumulative_pairs,
        max_variants_per_solution=args.max_variants,
        latent_dataset_name=args.dataset_name
    )