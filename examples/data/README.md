# AfCycDesign Example Data

This directory contains input data for AfCycDesign use case examples.

## Directory Structure

```
data/
├── README.md              # This file
├── structures/            # PDB structure files
├── sequences/             # Sequence files
├── test_backbone.pdb      # Simple test backbone
└── *.json                 # Configuration files
```

---

## Structure Files

### Benchmark Cyclic Peptides (for Structure Prediction - Use Case 0)

These are experimentally determined cyclic peptide structures from the AfCycDesign paper benchmarks.

| File | PDB ID | Name | Sequence | Length | Use Case |
|------|--------|------|----------|--------|----------|
| `1JBL.pdb` | 1JBL | SFTI-1 (Sunflower Trypsin Inhibitor) | GRCTKSIPPICFPD | 14 | Structure prediction benchmark |
| `1JBL_chainA.pdb` | 1JBL | SFTI-1 Chain A only | GRCTKSIPPICFPD | 14 | Fixbb design input |
| `2MW0.pdb` | 2MW0 | Kalata B7 Ser Mutant | FRLLNYYA | 8 | Structure prediction benchmark |
| `2LWV.pdb` | 2LWV | PDP-6 (PAWS-derived Peptide) | WTYTYDWFC | 9 | Structure prediction benchmark |
| `5KX1.pdb` | 5KX1 | NC_CHHH_D1 (Designed Peptide) | CWLPCFGDAC | 10 | Structure prediction benchmark |

### Binder Design Targets (for Binder Design - Use Case 3)

These are protein targets for cyclic peptide binder design.

| File | PDB ID | Name | Description | Chain | Use Case |
|------|--------|------|-------------|-------|----------|
| `4HFZ.pdb` | 4HFZ | MDM2-p53 Complex | E3 Ubiquitin Ligase | A (MDM2), B (p53 peptide) | Binder design target |
| `4HFZ_MDM2.pdb` | 4HFZ | MDM2 Chain A only | MDM2 binding domain | A | Clean binder target |
| `2FLU.pdb` | 2FLU | Keap1-Neh2 Complex | Kelch-like protein | X (Keap1), P (Nrf2 peptide) | Binder design target |
| `2FLU_Keap1.pdb` | 2FLU | Keap1 Chain X only | Keap1 Kelch domain | X | Clean binder target |

### Other Structures

| File | PDB ID | Description | Use Case |
|------|--------|-------------|----------|
| `1O91.pdb` | 1O91 | Collagen VIII NC1 Domain | General binder target |
| `1P3J.pdb` | 1P3J | Protein structure | Fixbb design input |
| `test_backbone.pdb` | - | Simple 3-residue cyclic backbone | Quick testing |

---

## Example Sequences

### Benchmark Sequences for Structure Prediction

From the AfCycDesign paper:

| PDB | Sequence | Length | Notes |
|-----|----------|--------|-------|
| 1JBL | `GRCTKSIPPICFPD` | 14 | Contains disulfide |
| 2MW0 | `FRLLNYYA` | 8 | Cyclotide mutant |
| 2LWV | `WTYTYDWFC` | 9 | PAWS-derived |
| 5KX1 | `CWLPCFGDAC` | 10 | Designed cyclic |

### Binding Motifs for Binder Design

| Target | Motif | Sequence | Notes |
|--------|-------|----------|-------|
| MDM2 | p53 hotspot | `FSDLW` | Key binding residues |
| Keap1 | Nrf2 hot loop | `DEETGE` | Critical interface |

---

## Usage Examples

### Structure Prediction (Use Case 0)

```bash
# Predict structure of benchmark sequence
python use_case_0_structure_prediction.py --sequence "GRCTKSIPPICFPD" --num_models 5 --num_recycles 6
```

### Fixed Backbone Design (Use Case 1)

```bash
# Redesign sequence for 1JBL backbone
python use_case_1_cyclic_fixbb_design.py --pdb data/structures/1JBL_chainA.pdb --chain A

# Use test backbone for quick testing
python use_case_1_cyclic_fixbb_design.py --pdb data/test_backbone.pdb --chain A
```

### Binder Design (Use Case 3)

```bash
# Design binder for MDM2
python use_case_3_cyclic_binder_design.py --pdb data/structures/4HFZ_MDM2.pdb --target_chain A --binder_len 14

# Design binder for Keap1
python use_case_3_cyclic_binder_design.py --pdb data/structures/2FLU_Keap1.pdb --target_chain X --binder_len 12
```

---

## Download Sources

All PDB files were downloaded from the RCSB Protein Data Bank:
- https://www.rcsb.org/structure/1JBL
- https://www.rcsb.org/structure/2MW0
- https://www.rcsb.org/structure/2LWV
- https://www.rcsb.org/structure/5KX1
- https://www.rcsb.org/structure/4HFZ
- https://www.rcsb.org/structure/2FLU

---

## References

- **1JBL**: Korsinczky et al. (2001) "Solution structures by 1H NMR of the novel cyclic trypsin inhibitor SFTI-1"
- **2MW0**: Wang et al. (2014) "Kalata B7 structure"
- **2LWV**: Gruber et al. (2012) "PAWS-derived peptide structures"
- **5KX1**: Bhardwaj et al. (2016) "De novo designed cyclic peptides"
- **4HFZ**: Michelsen et al. (2012) "MDM2-p53 peptide complex"
- **2FLU**: Lo et al. (2006) "Keap1-Neh2 complex structure"
