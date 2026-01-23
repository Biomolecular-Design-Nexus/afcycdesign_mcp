# AfCycDesign Quick Start Commands

Copy-paste commands for running each use case with the provided example data.

## Prerequisites

```bash
# Set environment variable (run once per session)
cd /path/to/afcycdesign_mcp
export ALPHAFOLD_DATA_DIR=./params

# Create output directory
mkdir -p examples/outputs
```

---

## Use Case 0: Structure Prediction

Predict 3D structure of known cyclic peptide sequences.

### Benchmark Sequences (from paper)

```bash
# 1JBL: SFTI-1 (14 residues) - Recommended settings
./env/bin/python examples/use_case_0_structure_prediction.py \
    --benchmark 1JBL \
    --num_models 5 \
    --num_recycles 6 \
    --output examples/outputs/1JBL_predicted.pdb \
    --save_json

# 2MW0: Kalata B7 (8 residues)
./env/bin/python examples/use_case_0_structure_prediction.py \
    --benchmark 2MW0 \
    --num_models 5 \
    --num_recycles 6 \
    --output examples/outputs/2MW0_predicted.pdb

# 2LWV: PDP-6 (9 residues)
./env/bin/python examples/use_case_0_structure_prediction.py \
    --benchmark 2LWV \
    --num_models 3 \
    --num_recycles 6 \
    --output examples/outputs/2LWV_predicted.pdb

# 5KX1: Designed peptide (10 residues)
./env/bin/python examples/use_case_0_structure_prediction.py \
    --benchmark 5KX1 \
    --num_models 3 \
    --num_recycles 6 \
    --output examples/outputs/5KX1_predicted.pdb
```

### Custom Sequence

```bash
# Your own sequence
./env/bin/python examples/use_case_0_structure_prediction.py \
    --sequence "RVKDGYPF" \
    --num_models 3 \
    --num_recycles 6 \
    --output examples/outputs/custom_predicted.pdb \
    --save_json
```

### Quick Test (reduced iterations)

```bash
./env/bin/python examples/use_case_0_structure_prediction.py \
    --sequence "FRLLNYYA" \
    --num_models 1 \
    --num_recycles 3 \
    --output examples/outputs/quick_test.pdb
```

---

## Use Case 1: Fixed Backbone Design

Redesign sequence for a cyclic peptide backbone structure.

### Using 1JBL Backbone

```bash
# Standard design (110-step schedule)
./env/bin/python examples/use_case_1_cyclic_fixbb_design.py \
    --pdb examples/data/structures/1JBL_chainA.pdb \
    --chain A \
    --stage_iters 50 50 10 \
    --output examples/outputs/1JBL_redesigned.pdb

# With compactness constraint
./env/bin/python examples/use_case_1_cyclic_fixbb_design.py \
    --pdb examples/data/structures/1JBL_chainA.pdb \
    --chain A \
    --add_rg \
    --rg_weight 0.1 \
    --stage_iters 50 50 10 \
    --output examples/outputs/1JBL_compact.pdb
```

### Quick Test

```bash
./env/bin/python examples/use_case_1_cyclic_fixbb_design.py \
    --pdb examples/data/test_backbone.pdb \
    --chain A \
    --stage_iters 20 20 5 \
    --output examples/outputs/test_redesigned.pdb
```

---

## Use Case 2: De Novo Hallucination

Generate novel cyclic peptide structure and sequence from scratch.

### Standard Hallucination

```bash
# 12-mer (paper-recommended settings)
./env/bin/python examples/use_case_2_cyclic_hallucination.py \
    --length 12 \
    --rm_aa "C" \
    --soft_iters 50 \
    --stage_iters 50 50 10 \
    --plddt_threshold 0.9 \
    --output examples/outputs/hallucinated_12mer.pdb

# 8-mer (quick)
./env/bin/python examples/use_case_2_cyclic_hallucination.py \
    --length 8 \
    --rm_aa "C" \
    --soft_iters 30 \
    --stage_iters 30 30 10 \
    --output examples/outputs/hallucinated_8mer.pdb
```

### Compact Structure (drug-like)

```bash
./env/bin/python examples/use_case_2_cyclic_hallucination.py \
    --length 10 \
    --rm_aa "C,M" \
    --add_rg \
    --rg_weight 0.15 \
    --soft_iters 50 \
    --stage_iters 50 50 10 \
    --plddt_threshold 0.9 \
    --output examples/outputs/compact_10mer.pdb
```

### Scaffold Library Generation

```bash
# Generate multiple scaffolds for binder design library
for i in {1..10}; do
    ./env/bin/python examples/use_case_2_cyclic_hallucination.py \
        --length 13 \
        --rm_aa "C" \
        --plddt_threshold 0.9 \
        --output examples/outputs/scaffold_${i}.pdb
done
```

---

## Use Case 3: Binder Design

Design cyclic peptide binders for protein targets.

### MDM2 Binder (Cancer Target)

```bash
# Basic MDM2 binder design
./env/bin/python examples/use_case_3_cyclic_binder_design.py \
    --pdb examples/data/structures/4HFZ_MDM2.pdb \
    --target_chain A \
    --binder_len 12 \
    --optimizer "pssm_semigreedy" \
    --num_models 2 \
    --ipae_threshold 0.15 \
    --output examples/outputs/MDM2_binder.pdb

# With p53-derived motif seed
./env/bin/python examples/use_case_3_cyclic_binder_design.py \
    --pdb examples/data/structures/4HFZ_MDM2.pdb \
    --target_chain A \
    --binder_seq "FSDLWKLLPEN" \
    --optimizer "3stage" \
    --ipae_threshold 0.11 \
    --output examples/outputs/MDM2_p53motif_binder.pdb
```

### Keap1 Binder (Oxidative Stress Target)

```bash
# Basic Keap1 binder design
./env/bin/python examples/use_case_3_cyclic_binder_design.py \
    --pdb examples/data/structures/2FLU_Keap1.pdb \
    --target_chain X \
    --binder_len 14 \
    --optimizer "pssm_semigreedy" \
    --num_models 2 \
    --ipae_threshold 0.15 \
    --output examples/outputs/Keap1_binder.pdb

# With Nrf2-derived motif
./env/bin/python examples/use_case_3_cyclic_binder_design.py \
    --pdb examples/data/structures/2FLU_Keap1.pdb \
    --target_chain X \
    --binder_seq "DEETGEFLDEET" \
    --optimizer "3stage" \
    --output examples/outputs/Keap1_nrf2_binder.pdb
```

### General Target

```bash
# Design binder for any target protein
./env/bin/python examples/use_case_3_cyclic_binder_design.py \
    --pdb examples/data/structures/1O91.pdb \
    --target_chain A \
    --binder_len 14 \
    --hotspot "20-30,45-55" \
    --optimizer "pssm_semigreedy" \
    --output examples/outputs/1O91_binder.pdb
```

---

## GPU Usage

Add `--gpu N` to any command to use GPU N:

```bash
# Use GPU 0
./env/bin/python examples/use_case_2_cyclic_hallucination.py \
    --length 12 \
    --output examples/outputs/gpu_test.pdb \
    --gpu 0

# Use GPU 1 with memory limit
./env/bin/python examples/use_case_2_cyclic_hallucination.py \
    --length 12 \
    --output examples/outputs/gpu_test.pdb \
    --gpu 1 \
    --gpu_mem_fraction 0.8
```

---

## Using YAML Configs

```bash
# Structure prediction with YAML config
./env/bin/python scripts/predict_cyclic_structure.py \
    --config examples/use_case_2_hallucination.yaml \
    --gpu 0

# Sequence prediction with YAML
./env/bin/python scripts/predict_cyclic_structure.py \
    --config examples/use_case_0_structure_prediction.yaml \
    --gpu 0
```

---

## Run All Examples Script

```bash
# Run all use cases interactively
./examples/run_all_use_cases.sh help

# Run specific use case
./examples/run_all_use_cases.sh 0 0    # Use Case 0 on GPU 0
./examples/run_all_use_cases.sh 2 1    # Use Case 2 on GPU 1
./examples/run_all_use_cases.sh all 0  # All use cases on GPU 0
```

---

## Expected Output Quality

| Use Case | Metric | Good | Excellent |
|----------|--------|------|-----------|
| 0 (Structure Prediction) | pLDDT | > 0.70 | > 0.90 |
| 1 (Fixbb Design) | pLDDT | > 0.70 | > 0.85 |
| 2 (Hallucination) | pLDDT | > 0.70 | > 0.90 |
| 3 (Binder Design) | iPAE | < 0.15 | < 0.11 |
