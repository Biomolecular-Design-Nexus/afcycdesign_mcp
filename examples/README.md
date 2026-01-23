# AfCycDesign Examples

Example scripts and configurations for cyclic peptide structure prediction and design using AlphaFold.

**Reference:** Stephen Rettie et al., doi: https://doi.org/10.1101/2023.02.25.529956

## Table of Contents

- [Overview of Use Cases](#overview-of-use-cases)
- [Use Case 0: Structure Prediction](#use-case-0-structure-prediction)
- [Use Case 1: Fixed Backbone Design (Fixbb)](#use-case-1-fixed-backbone-design-fixbb)
- [Use Case 2: De Novo Hallucination](#use-case-2-de-novo-hallucination)
- [Use Case 3: Binder Design (Motif Grafting)](#use-case-3-binder-design-motif-grafting)
- [Configuration Files](#configuration-files)
- [Quality Thresholds](#quality-thresholds)

---

## Overview of Use Cases

| Use Case | Description | Input | Key Output |
|----------|-------------|-------|------------|
| **0. Structure Prediction** | Predict 3D structure of known cyclic peptide sequence | Amino acid sequence | 3D structure (PDB) |
| **1. Fixed Backbone Design** | Find optimal sequence for a given backbone structure | Target backbone PDB | Designed sequence |
| **2. De Novo Hallucination** | Generate novel cyclic peptide structure + sequence | Peptide length | New scaffold |
| **3. Binder Design** | Design cyclic peptide binders for protein targets | Target protein PDB | Binder sequence + complex |

---

## Use Case 0: Structure Prediction

**Purpose:** Predict the 3D structure of a known cyclic peptide sequence.

### Key Parameters (from paper)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Recycles | 6 | Number of AlphaFold recycles |
| Models | 5 | Evaluate all 5 AF2 models |
| Features | Single Sequence or MSA | With random masking |
| Cyclic Offset | N x N matrix | Terminal separation = 1 or -1 |
| Quality Filter | pLDDT > 0.7 | High confidence threshold |

### Benchmark Sequences (from paper)

| PDB ID | Sequence | Length |
|--------|----------|--------|
| 1JBL | GFNYGPFGSC | 10 |
| 2MW0 | FRLLNYYA | 8 |
| 2LWV | WTYTYDWFC | 9 |
| 5KX1 | CWLPCFGDAC | 10 |

### Usage

```bash
# Basic structure prediction
python use_case_0_structure_prediction.py --sequence "GFNYGPFGSC" --output predicted.pdb

# Paper-recommended settings (all 5 models, 6 recycles)
python use_case_0_structure_prediction.py --sequence "FRLLNYYA" --num_models 5 --num_recycles 6

# Use benchmark sequence directly
python use_case_0_structure_prediction.py --benchmark 1JBL --save_json
```

### Workflow

1. Apply custom N x N cyclic offset matrix to relative positional encoding
2. Define sequence separation between terminal residues as 1 or -1
3. Run prediction across all specified AlphaFold models
4. Filter results by pLDDT (> 0.7 = high confidence)

---

## Use Case 1: Fixed Backbone Design (Fixbb)

**Purpose:** Find a sequence that maximizes folding propensity for a pre-defined backbone.

### Key Parameters (from paper)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Optimization | 110 steps | Stage 1 (50) + Stage 2 (50) + Stage 3 (10) |
| Loss Function | CCE | Categorical cross-entropy (target vs predicted distogram) |
| Optimizer | Straight-through | Continuous logits -> one-hot encoding |

### Usage

```bash
# Basic fixed backbone design
python use_case_1_cyclic_fixbb_design.py --pdb data/test_backbone.pdb --chain A --output designed.pdb

# Paper-recommended 110-step schedule
python use_case_1_cyclic_fixbb_design.py --pdb input.pdb --stage_iters 50 50 10

# With compactness constraint
python use_case_1_cyclic_fixbb_design.py --pdb input.pdb --add_rg --rg_weight 0.1
```

### Workflow

1. Extract distogram from target PDB
2. Initialize with random sequence
3. Iteratively optimize sequence to minimize CCE loss
4. Transition from continuous logits to one-hot using straight-through estimator

---

## Use Case 2: De Novo Hallucination

**Purpose:** Simultaneously generate a new cyclic peptide structure AND sequence from scratch.

### Key Parameters (from paper)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Input | Length only | 7-16 residues successfully tested |
| Initialization | Random Gumbel | Gumbel distribution |
| Loss Function | 1 - pLDDT + PAE/31 + con/2 | Combined loss |
| Contact Loss | binary=True, cutoff=21.6875, seqsep=0 | Contact settings |
| Temperature | 1.0 -> 0.01 | Softmax annealing |
| Optimization | 50, 50, 10 steps | 3-stage protocol |
| Quality Filter | pLDDT > 0.9 | High-confidence scaffolds |

### Usage

```bash
# Basic hallucination
python use_case_2_cyclic_hallucination.py --length 13 --output scaffold.pdb

# Paper-recommended settings for scaffold library
python use_case_2_cyclic_hallucination.py --length 13 --rm_aa "C" --plddt_threshold 0.9

# Compact structure with Rg constraint
python use_case_2_cyclic_hallucination.py --length 15 --add_rg --rg_weight 0.15
```

### Workflow

1. Run 3-stage optimization protocol (50, 50, 10 steps)
2. Enable cyclic offset specifically for sequence optimization
3. Filter resulting scaffolds for pLDDT > 0.9

**Note:** The authors generated ~24,000 high-confidence scaffolds for their library using this workflow.

---

## Use Case 3: Binder Design (Motif Grafting)

**Purpose:** Design cyclic peptide binders for protein targets (e.g., MDM2, Keap1).

This is the most complex workflow, combining multiple tools and filtering steps.

### Input Files (from paper)

| Input | Example | Description |
|-------|---------|-------------|
| Target Protein | MDM2 (4HFZ), Keap1 (2FLU) | PDB structure |
| Binding Motif | p53: FSDLW, Nrf2: DEETGE | Functional segments |
| Scaffold Library | ~24,000 scaffolds | Hallucinated peptides with pLDDT > 0.9 |

### Key Parameters (from paper)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Grafting | 1.0 A RMSD | Rosetta MotifGraft tolerance |
| Sequence Design | 3-4 rounds | ProteinMPNN + Rosetta (REF2015) |
| Cyclic Offset | Binder only | NOT applied to target chain |

### Filtering Thresholds (from paper)

| Metric | Threshold | Description |
|--------|-----------|-------------|
| iPAE | < 0.15 (or 0.11 strict) | Interface Predicted Aligned Error |
| ddG | < -30 kcal/mol | Rosetta binding energy |
| SAP | < 30 | Spatial Aggregation Propensity |
| CMS | > 300 | Contact Molecular Surface |
| RMSD | < 1.5 A | Binding mode RMSD |

### Usage

```bash
# Basic binder design
python use_case_3_cyclic_binder_design.py --pdb 4HFZ --target_chain A --binder_len 14 --output mdm2_binder.pdb

# Target specific hotspot residues
python use_case_3_cyclic_binder_design.py --pdb 2FLU --target_chain A --hotspot "20-30" --binder_len 12

# With initial motif sequence
python use_case_3_cyclic_binder_design.py --pdb 4HFZ --binder_seq "FSDLWKLLPEN" --output motif_binder.pdb

# Strict iPAE filtering
python use_case_3_cyclic_binder_design.py --pdb 4HFZ --binder_len 14 --ipae_threshold 0.11
```

### Workflow

1. Graft functional motif onto stable hallucinated scaffolds
2. Redesign non-motif residues using ProteinMPNN
3. Use AfCycDesign to predict target-peptide complex (cyclic offset on binder ONLY)
4. Verify binding mode via iPAE and RMSD < 1.5 A

---

## Configuration Files

### YAML Configurations

| File | Use Case | Description |
|------|----------|-------------|
| `use_case_0_structure_prediction.yaml` | Structure Prediction | Predict structure from sequence |
| `use_case_1_fixbb_design.yaml` | Fixbb Design | Redesign sequence for backbone |
| `use_case_2_hallucination.yaml` | Hallucination | Generate novel scaffold |
| `use_case_3_binder_design.yaml` | Binder Design | Design protein binder |

### Quick Test Configurations

| File | Description |
|------|-------------|
| `predict_8mer.yaml` | Quick 8-mer hallucination test |
| `predict_12mer_production.yaml` | Production 12-mer with more iterations |
| `predict_compact_peptide.yaml` | Compact structure with Rg constraint |
| `predict_from_sequence.yaml` | Predict structure from known sequence |

### Usage with YAML

```bash
# Using main script with YAML config
ALPHAFOLD_DATA_DIR=./params python scripts/predict_cyclic_structure.py --config examples/use_case_2_hallucination.yaml --gpu 0
```

---

## Quality Thresholds

### Structure Prediction / Hallucination

| Metric | Excellent | Good | Acceptable |
|--------|-----------|------|------------|
| pLDDT | > 0.90 | > 0.70 | > 0.50 |
| PAE | < 0.10 | < 0.30 | < 0.50 |

### Binder Design

| Metric | Threshold | Notes |
|--------|-----------|-------|
| iPAE | < 0.15 | < 0.11 for strict filtering |
| ddG | < -30 kcal/mol | Rosetta binding energy |
| SAP | < 30 | Aggregation propensity |
| CMS | > 300 | Contact surface |
| RMSD | < 1.5 A | Binding mode accuracy |

---

## Directory Structure

```
examples/
├── README.md                           # This file
├── data/
│   ├── structures/                     # Sample PDB structures
│   │   ├── 1O91.pdb                   # Binder design target
│   │   └── 1P3J.pdb                   # Fixbb design target
│   ├── sequences/                      # Sample sequences
│   └── test_backbone.pdb              # Test backbone for fixbb
├── use_case_0_structure_prediction.py  # Structure prediction script
├── use_case_0_structure_prediction.yaml
├── use_case_1_cyclic_fixbb_design.py   # Fixed backbone design script
├── use_case_1_fixbb_design.yaml
├── use_case_2_cyclic_hallucination.py  # Hallucination script
├── use_case_2_hallucination.yaml
├── use_case_3_cyclic_binder_design.py  # Binder design script
├── use_case_3_binder_design.yaml
├── predict_8mer.yaml                   # Quick test config
├── predict_12mer_production.yaml       # Production config
├── predict_compact_peptide.yaml        # Compact structure config
├── predict_from_sequence.yaml          # Sequence prediction config
└── run_example.sh                      # Runner script
```

---

## Quick Start Examples

### 1. Predict structure of a known sequence

```bash
python use_case_0_structure_prediction.py --benchmark 1JBL --num_models 5 --num_recycles 6 --output benchmark_1JBL.pdb
```

### 2. Generate a novel cyclic peptide scaffold

```bash
python use_case_2_cyclic_hallucination.py --length 12 --rm_aa "C" --plddt_threshold 0.9 --output novel_scaffold.pdb
```

### 3. Redesign sequence for existing backbone

```bash
python use_case_1_cyclic_fixbb_design.py --pdb data/test_backbone.pdb --chain A --stage_iters 50 50 10 --output redesigned.pdb
```

### 4. Design a binder for MDM2

```bash
python use_case_3_cyclic_binder_design.py --pdb_code 4HFZ --target_chain A --binder_len 14 --hotspot "20-30" --ipae_threshold 0.15 --output mdm2_cyclic_binder.pdb
```

---

## GPU Usage

All scripts support GPU acceleration:

```bash
# Basic GPU usage
python use_case_X.py ... --gpu 0

# Limit GPU memory
python use_case_X.py ... --gpu 0 --gpu_mem_fraction 0.8

# Force CPU
python use_case_X.py ... --cpu
```

---

## References

- **AfCycDesign Paper:** Rettie et al., "Cyclic peptide structure prediction and design using AlphaFold"
- **ColabDesign:** https://github.com/sokrypton/ColabDesign
- **AlphaFold:** Jumper et al., Nature 2021
