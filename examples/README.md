# Cyclic Peptide Design Examples

Example configurations and data for reproducing cyclic peptide structure predictions.

## Directory Structure

```
example/
├── README.md
├── data/
│   ├── config_quick_test.json      # Fast testing config (reduced iterations)
│   ├── config_production.json      # High-quality production config
│   ├── config_compact_peptide.json # Compact structure with Rg constraint
│   └── example_8mer.pdb            # Example output structure
└── outputs/                        # Generated outputs go here
```

## Quick Start

### 1. Basic 8-mer prediction (CPU)

```bash
cd /path/to/tool-mcps/afcycdesign_mcp
ALPHAFOLD_DATA_DIR=./params ./env/bin/python scripts/predict_cyclic_structure.py \
  --length 8 \
  --output example/outputs/test_8mer.pdb \
  --config example/data/config_quick_test.json
```

### 2. GPU-accelerated prediction

```bash
# Use GPU 0
ALPHAFOLD_DATA_DIR=./params ./env/bin/python scripts/predict_cyclic_structure.py \
  --length 8 \
  --output example/outputs/gpu_8mer.pdb \
  --gpu 0

# Use GPU 1 with memory limit
ALPHAFOLD_DATA_DIR=./params ./env/bin/python scripts/predict_cyclic_structure.py \
  --length 12 \
  --output example/outputs/gpu_12mer.pdb \
  --gpu 1 \
  --gpu_mem_fraction 0.8
```

### 3. Compact peptide with Rg constraint

```bash
ALPHAFOLD_DATA_DIR=./params ./env/bin/python scripts/predict_cyclic_structure.py \
  --length 10 \
  --output example/outputs/compact_10mer.pdb \
  --config example/data/config_compact_peptide.json \
  --gpu 0
```

### 4. Production-quality prediction

```bash
ALPHAFOLD_DATA_DIR=./params ./env/bin/python scripts/predict_cyclic_structure.py \
  --length 8 \
  --output example/outputs/production_8mer.pdb \
  --config example/data/config_production.json \
  --gpu 0
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `rm_aa` | Amino acids to exclude (comma-separated) | `"C"` |
| `offset_type` | Cyclic offset type (1, 2, or 3) | `2` |
| `add_rg` | Add radius of gyration constraint | `false` |
| `rg_weight` | Weight for Rg loss term | `0.1` |
| `num_recycles` | Number of AlphaFold recycles | `0` |
| `soft_iters` | Soft pre-optimization iterations | `50` |
| `stage_iters` | 3-stage iterations [logits, soft, hard] | `[50, 50, 10]` |
| `contact_cutoff` | Contact distance cutoff | `21.6875` |
| `loss_weights` | Weights for pae, plddt, con losses | `{pae:1, plddt:1, con:0.5}` |

## GPU Options

| Option | Description |
|--------|-------------|
| `--gpu ID` | GPU device ID (0, 1, etc.) |
| `--gpu_mem_fraction FRAC` | Fraction of GPU memory to use (0.0-1.0) |
| `--no_gpu_preallocate` | Disable GPU memory preallocation |
| `--cpu` | Force CPU mode |

## Expected Output

Each run produces:
1. **PDB file**: 3D structure of the cyclic peptide
2. **JSON file**: Metadata including sequence, metrics, and configuration

Example metrics:
- **pLDDT**: Predicted Local Distance Difference Test (>0.70 is good, >0.90 is excellent)
- **PAE**: Predicted Aligned Error (<0.30 is good, <0.10 is excellent)
