# Cyclic Peptide Design Examples

Example configurations and data for reproducing cyclic peptide structure predictions.

## Two Prediction Modes

The script supports two modes:

1. **Hallucination Mode**: Generate both sequence AND structure from scratch
   - Use when you want to design a new cyclic peptide
   - Specify `peptide.length` in config or `--length` on CLI

2. **Sequence Mode**: Predict structure for a GIVEN sequence
   - Use when you have a known sequence and want its 3D structure
   - Specify `peptide.sequence` in config or `--sequence` on CLI
   - Sequence is preserved exactly (no design changes)

## Directory Structure

```
example/
├── README.md
├── run_example.sh                  # Convenience runner script
├── data/
│   ├── predict_8mer.yaml           # Quick 8-mer hallucination (YAML)
│   ├── predict_12mer_production.yaml  # Production 12-mer hallucination (YAML)
│   ├── predict_compact_peptide.yaml   # Compact with Rg constraint (YAML)
│   ├── predict_from_sequence.yaml  # Structure prediction from sequence (YAML)
│   ├── config_quick_test.json      # Fast testing (JSON, legacy)
│   ├── config_production.json      # Production config (JSON, legacy)
│   ├── config_compact_peptide.json # Compact config (JSON, legacy)
│   ├── example_8mer.pdb            # Example output structure
│   └── example_8mer.json           # Example output metadata
└── outputs/                        # Generated outputs go here
```

## Quick Start with YAML Configs (Recommended)

YAML configs provide a clean, readable way to specify all prediction parameters:

### 1. Basic 8-mer hallucination (generate sequence + structure)

```bash
cd /path/to/tool-mcps/afcycdesign_mcp
ALPHAFOLD_DATA_DIR=./params ./env/bin/python scripts/predict_cyclic_structure.py \
  --config example/data/predict_8mer.yaml \
  --gpu 0
```

### 2. Structure prediction from sequence (head-to-tail cyclization)

```bash
ALPHAFOLD_DATA_DIR=./params ./env/bin/python scripts/predict_cyclic_structure.py \
  --config example/data/predict_from_sequence.yaml \
  --gpu 0
```

### 3. Production 12-mer hallucination

```bash
ALPHAFOLD_DATA_DIR=./params ./env/bin/python scripts/predict_cyclic_structure.py \
  --config example/data/predict_12mer_production.yaml \
  --gpu 1
```

### 4. Compact peptide with Rg constraint

```bash
ALPHAFOLD_DATA_DIR=./params ./env/bin/python scripts/predict_cyclic_structure.py \
  --config example/data/predict_compact_peptide.yaml \
  --gpu 0
```

## YAML Config Format

### Hallucination Mode (design sequence + structure)

```yaml
# Job identification
name: "my_peptide_job"
description: "Description of the prediction"

# Peptide specification - use LENGTH for hallucination
peptide:
  length: 8                      # Number of residues (5-50)
  exclude_amino_acids: "C,M"     # Amino acids to exclude from design

# Structure constraints
constraints:
  add_rg: true                   # Radius of gyration constraint
  rg_weight: 0.15                # Rg loss weight
  offset_type: 2                 # Cyclic offset type

# Optimization parameters
optimization:
  num_recycles: 0                # AlphaFold recycles
  soft_iters: 50                 # Soft optimization iterations
  stage_iters: [50, 50, 10]      # 3-stage iterations

# Loss function weights
loss_weights:
  pae: 1.0
  plddt: 1.0
  con: 0.5

# Output settings
output:
  file: "outputs/my_peptide.pdb" # Optional, auto-generated if not specified
  save_metadata: true

# GPU settings
gpu:
  device: 0                      # GPU ID (null for auto, -1 for CPU)
  mem_fraction: 0.9              # Memory fraction (0.0-1.0)
```

### Sequence Mode (predict structure for given sequence)

```yaml
# Job identification
name: "structure_prediction"
description: "Predict structure for given cyclic peptide sequence"

# Peptide specification - use SEQUENCE for structure prediction
peptide:
  sequence: "RVKDGYPF"           # Natural amino acids only (A-Y, no X/B/Z)

# Structure constraints (head-to-tail cyclization applied automatically)
constraints:
  add_rg: false                  # Optional compactness constraint
  offset_type: 2                 # Cyclic offset type

# Optimization parameters
optimization:
  num_recycles: 1                # More recycles can improve prediction

# Output settings
output:
  save_metadata: true

# GPU settings
gpu:
  device: 0
```

## Command-Line Usage (Alternative)

You can also specify parameters directly via command line:

### Hallucination mode (design new peptide)

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

### Sequence mode (predict structure for given sequence)

```bash
# Predict structure for a specific cyclic peptide sequence
ALPHAFOLD_DATA_DIR=./params ./env/bin/python scripts/predict_cyclic_structure.py \
  --sequence "RVKDGYPF" \
  --output example/outputs/rvkdgypf_structure.pdb \
  --gpu 0

# With compactness constraint
ALPHAFOLD_DATA_DIR=./params ./env/bin/python scripts/predict_cyclic_structure.py \
  --sequence "ACDEFGHIKLMN" \
  --output example/outputs/12mer_structure.pdb \
  --add_rg \
  --gpu 1
```

### Compact peptide with Rg constraint (hallucination)

```bash
ALPHAFOLD_DATA_DIR=./params ./env/bin/python scripts/predict_cyclic_structure.py \
  --length 10 \
  --output example/outputs/compact_10mer.pdb \
  --add_rg \
  --rg_weight 0.15 \
  --rm_aa "C,M" \
  --gpu 0
```

## Configuration Options

| Parameter | YAML Path | CLI Flag | Default | Description |
|-----------|-----------|----------|---------|-------------|
| Length | `peptide.length` | `--length` | - | Peptide length (5-50) for hallucination mode |
| Sequence | `peptide.sequence` | `--sequence` | - | Amino acid sequence for structure prediction |
| Exclude AA | `peptide.exclude_amino_acids` | `--rm_aa` | `"C"` | Amino acids to exclude (hallucination only) |
| Add Rg | `constraints.add_rg` | `--add_rg` | `false` | Compactness constraint |
| Rg Weight | `constraints.rg_weight` | `--rg_weight` | `0.1` | Rg loss weight |
| Recycles | `optimization.num_recycles` | `--num_recycles` | `0` | AF recycles |
| Soft Iters | `optimization.soft_iters` | `--soft_iters` | `50` | Soft optimization iters |
| Stage Iters | `optimization.stage_iters` | `--stage_iters` | `[50,50,10]` | 3-stage iterations |

**Note:** Provide either `length` (hallucination mode) OR `sequence` (structure prediction mode), not both.

## GPU Options

| Option | YAML Path | CLI Flag | Description |
|--------|-----------|----------|-------------|
| Device ID | `gpu.device` | `--gpu` | GPU device (0, 1, etc.) |
| Memory | `gpu.mem_fraction` | `--gpu_mem_fraction` | Memory fraction (0.0-1.0) |
| CPU mode | - | `--cpu` | Force CPU mode |

## Expected Output

Each run produces:
1. **PDB file**: 3D structure of the cyclic peptide
2. **JSON file**: Metadata including sequence, metrics, and configuration

Example metrics:
- **pLDDT**: Predicted Local Distance Difference Test (>0.70 is good, >0.90 is excellent)
- **PAE**: Predicted Aligned Error (<0.30 is good, <0.10 is excellent)

## Convenience Script

Use the runner script for quick testing:

```bash
./run_example.sh quick 0    # Quick test on GPU 0
./run_example.sh production 1  # Production on GPU 1
./run_example.sh compact 0   # Compact peptide on GPU 0
```
