# ProteusQUBO: Quantum-Inspired Molecular Design Framework

A QUBO (Quadratic Unconstrained Binary Optimization) based molecular design and optimization system for drug discovery and molecular property optimization.

## Overview

ProteusQUBO combines deep learning and quantum-inspired optimization techniques to enable efficient molecular design in continuous latent spaces. The framework consists of three main components:

- **Block_bAE**: Molecular autoencoder for bidirectional SMILES ↔ binary latent vector conversion
- **Qmol_FM**: Factorization machine-based optimizer with QUBO formulation
- **RBM**: Restricted Boltzmann Machine for molecular sampling

## Key Features

- **Binary Latent Space**: Discrete optimization-friendly molecular representation
- **QUBO/HUBO Optimization**: Quantum-inspired optimization for molecular properties
- **Multi-Objective Design**: Simultaneous optimization of multiple molecular properties
- **Scalable Architecture**: Parallel processing support for large-scale molecular screening

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ProteusQUBO.git
cd ProteusQUBO

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Encode SMILES to Latent Vectors

```python
from Block_bAE.inference_latent import main as encode_smiles

# Convert SMILES strings to binary latent vectors
encode_smiles(
    tsv_path="input_molecules.tsv",
    trained_model_ckpt_path="path/to/model.ckpt",
    output_path="latent_codes.h5"
)
```

### 2. Optimize Latent Vectors

```python
from Qmol_FM.opt import optimize_solutions_batch

# Optimize molecular properties using QUBO solver
optimize_solutions_batch(
    qubo_csv_path="fm_qubo_matrix.csv",
    input_h5_path="latent_codes.h5",
    output_h5_path="optimized_codes.h5",
    single_features_json="single_features.json",
    pair_configs_json="pair_configs.json",
    n_jobs=8
)
```

### 3. Decode to SMILES

```python
from Block_bAE.latent_to_smiles import main as decode_latent

# Convert optimized latent vectors back to SMILES
decode_latent(
    input_h5="optimized_codes.h5",
    model_ckpt_path="path/to/model.ckpt",
    vocab_path="vocab.json",
    output_h5="output_molecules.h5"
)
```

## Architecture

### Block_bAE (Binary Autoencoder)

- **Encoder**: GRU-based sequential encoder with Gumbel-Softmax sampling
- **Latent Space**: Binary vectors (128/328/628 dimensions)
- **Decoder**: Transformer-based decoder with attention mechanism
- **Training**: Reconstruction loss + KL divergence regularization

### Qmol_FM (Optimization Module)

- **Factorization Machine**: Learns property-bit interactions from training data
- **QUBO Formulation**: Converts FM to quadratic/higher-order unconstrained binary optimization
- **Optimization Strategies**:
  - Single-bit flip optimization
  - Pairwise configuration optimization
  - Greedy cumulative optimization
- **Parallel Processing**: Multi-core support for batch optimization

### RBM (Sampling Module)

- **Gibbs Sampling**: Generate novel molecular latent codes
- **Batch Processing**: Memory-efficient sampling for large-scale generation
- **Energy-Based Model**: Learn molecular distribution from training set

## Project Structure

```
ProteusQUBO_GitHub/
├── Block_bAE/                          # Molecular autoencoder
│   ├── model_gru_transformer.py        # Model architecture
│   ├── inference_latent.py             # SMILES → Latent encoding
│   ├── latent_to_smiles.py             # Latent → SMILES decoding
│   ├── train_gruencoder_transformerdecoder.py  # Training script
│   └── datamodule.py                   # Data loader
├── Qmol_FM/                            # Optimization module
│   ├── opt.py                          # Multi-variant optimizer (main)
│   ├── convert_fm_to_qubo.py           # FM → QUBO conversion
│   ├── convert_ising_to_qubo.py        # Ising → QUBO conversion
│   ├── sample.py                       # Ising solver interface
│   ├── utils_Qmol_FM.py                # Utility functions
│   └── hubo.py                         # Higher-order QUBO support
├── RBM/                                # Sampling module
│   ├── train_rbm.py                    # RBM model definition
│   └── sample_from_rbm.py              # Sampling script
├── .gitignore                          # Git ignore rules
├── README.md                           # This file
└── requirements.txt                    # Python dependencies
```

## Dependencies

- **Deep Learning**: PyTorch ≥ 2.0, PyTorch Lightning ≥ 2.0
- **Molecular Processing**: RDKit, SELFIES
- **Data Processing**: NumPy, Pandas, H5PY
- **Machine Learning**: scikit-learn, LightGBM
- **Optimization**: Kaiwu SDK (optional, for quantum annealing hardware)

See `requirements.txt` for complete dependency list.

## Usage Notes

### Trained Models Required

This repository contains **model architecture code only**. To run inference or training, you will need:

1. **Pre-trained autoencoder checkpoint** (`.ckpt` file)
2. **Vocabulary file** (`vocab.json`)
3. **Training dataset** (for optimization and RBM)

These files are not included due to size constraints and proprietary considerations.

### Quantum Solver Configuration

The `Qmol_FM/sample.py` module interfaces with Kaiwu quantum annealing platform. To use:

```bash
python Qmol_FM/sample.py \
    --ising_csv input_ising.csv \
    --output_h5 solutions.h5 \
    --output_log solver.log \
    --user_id YOUR_USER_ID \
    --sdk_code YOUR_SDK_CODE \
    --num_solutions 500
```

**Note**: `user_id` and `sdk_code` are required credentials for accessing the quantum solver service.

## Citation

If you use this code in your research, please cite:

```
[Citation information to be added upon publication]
```

## License

[To be determined - add license information]

## Contact

For questions regarding this codebase, please contact:
- [Your contact information]

---

**Disclaimer**: This repository is for academic research and peer review purposes. Model weights and training data are not included.
