# AfCycDesign Examples

This directory contains standalone Python scripts demonstrating cyclic peptide structure prediction and design using AlphaFold (AfCycDesign).

## Use Cases

### UC-001: Cyclic Peptide Fixed Backbone Design
**Script**: `use_case_1_cyclic_fixbb_design.py`
**Description**: Redesigns amino acid sequences for known cyclic peptide backbone structures while maintaining head-to-tail cyclization.

**Example Usage**:
```bash
# Design sequence for PDB structure 7m28 chain A
python use_case_1_cyclic_fixbb_design.py --pdb_code 7m28 --chain A --output designed_cyclic.pdb

# Use local PDB file with radius of gyration constraint
python use_case_1_cyclic_fixbb_design.py --pdb data/structures/1P3J.pdb --chain A --add_rg --output compact_design.pdb
```

### UC-002: Cyclic Peptide Hallucination
**Script**: `use_case_2_cyclic_hallucination.py`
**Description**: Generates novel cyclic peptide structures from scratch for a given length with head-to-tail cyclization constraints.

**Example Usage**:
```bash
# Generate 13-residue cyclic peptide (no cysteines)
python use_case_2_cyclic_hallucination.py --length 13 --output hallucinated_13mer.pdb

# Generate compact 15-residue peptide with RG constraint, excluding cysteine and methionine
python use_case_2_cyclic_hallucination.py --length 15 --rm_aa "C,M" --add_rg --output compact_15mer.pdb
```

### UC-003: Cyclic Peptide Binder Design
**Script**: `use_case_3_cyclic_binder_design.py`
**Description**: Designs cyclic peptide binders that bind to target protein structures while maintaining cyclization.

**Example Usage**:
```bash
# Design 14-residue cyclic binder for protein 4N5T chain A
python use_case_3_cyclic_binder_design.py --pdb_code 4N5T --target_chain A --binder_len 14 --output cyclic_binder.pdb

# Design with specific hotspot targeting
python use_case_3_cyclic_binder_design.py --pdb data/structures/1O91.pdb --target_chain A --binder_len 12 --hotspot "1-10,15" --output specific_binder.pdb
```

## Demo Data

### Structures (`data/structures/`)
- `1O91.pdb` - Example protein structure for target binding
- `1P3J.pdb` - Example protein structure for backbone design

### Sequences (`data/sequences/`)
- `sample_cyclic_peptides.txt` - Example cyclic peptide sequences with annotations

## Requirements

All scripts require the conda environment with ColabDesign installed. Activate the environment before running:

```bash
# Activate the environment
conda activate ./env

# Run any example script
python use_case_1_cyclic_fixbb_design.py --help
```

## Common Parameters

### General Options
- `--output` - Output PDB file path
- `--quiet` - Suppress verbose output
- `--num_recycles` - Number of AlphaFold recycles (0-6, default: 0)

### Cyclic Constraints
- `--offset_type` - Type of cyclic offset (1, 2, or 3, default: 2)
- `--add_rg` - Add radius of gyration loss for compact structures
- `--rg_weight` - Weight for RG loss (default: 0.1)

### Sequence Control
- `--rm_aa` - Amino acids to exclude (e.g., "C" or "C,M")

## Expected Outputs

Each script produces:
1. **PDB file** - 3D structure of the designed cyclic peptide
2. **Sequences** - Designed amino acid sequences
3. **Metrics** - Quality metrics (pLDDT, PAE, contacts, etc.)

## Troubleshooting

### Memory Issues
If you encounter GPU memory issues:
- Reduce `--num_models` parameter
- Use `--num_recycles 0`
- Try shorter peptide lengths

### AlphaFold Weights
Scripts may download AlphaFold weights on first run (~2.3GB). Ensure sufficient disk space and internet connection.

### CUDA Issues
If JAX CUDA installation is incomplete, the scripts will fall back to CPU mode (slower but functional).

## References

- **AfCycDesign Paper**: Stephen Rettie et al., "Cyclic peptide structure prediction and design using AlphaFold", doi: https://doi.org/10.1101/2023.02.25.529956
- **ColabDesign**: https://github.com/sokrypton/ColabDesign