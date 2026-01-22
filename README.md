# AfCycDesign MCP

> MCP tools for cyclic peptide computational analysis and design using AlphaFold

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Local Usage (Scripts)](#local-usage-scripts)
- [MCP Server Installation](#mcp-server-installation)
- [Using with Claude Code](#using-with-claude-code)
- [Using with Gemini CLI](#using-with-gemini-cli)
- [Available Tools](#available-tools)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

This MCP server provides computational tools for cyclic peptide structure prediction, sequence design, and binder development using the ColabDesign framework. The server offers both fast synchronous operations and long-running asynchronous jobs for comprehensive cyclic peptide research workflows.

### Features
- **3D Structure Prediction**: Predict cyclic peptide using AlphaFold2
- **Sequence Design**: Redesign amino acid sequences for given cyclic backbone structures
- **Binder Development**: Design cyclic peptide binders that bind to target protein structures
- **Batch Processing**: Generate multiple peptides with different parameters in parallel
- **Job Management**: Track long-running computations with status monitoring and log access
- **Quality Assessment**: Comprehensive structural quality metrics (pLDDT, PAE, contacts)


## Installation

### Quick Setup

Run the automated setup script:

```bash
./quick_setup.sh
```

This will create the environment and install all dependencies automatically.

Please view [`quick_setup.sh`](./quick_setup.sh) for manuall install.

## Local Usage (Scripts)

You can use the scripts directly without MCP for local processing.

### Available Scripts

| Script | Description | Example Use Case |
|--------|-------------|------------------|
| `scripts/predict_cyclic_structure.py` | Predict 3D structure from for a cyclic peptide | Novel peptide design |
| `scripts/design_cyclic_sequence.py` | Redesign sequence for given backbone | Optimize existing structures |
| `scripts/design_cyclic_binder.py` | Design binders to target proteins | Drug development |

### Script Examples

#### Predict 3D Structure

```bash

# Structure prediction from sequence (head-to-tail cyclization)
ALPHAFOLD_DATA_DIR=./params ./env/bin/python scripts/predict_cyclic_structure.py --config examples/predict_from_sequence.yaml --gpu 0

# Production 12-mer hallucination
ALPHAFOLD_DATA_DIR=./params ./env/bin/python scripts/predict_cyclic_structure.py --config examples/predict_12mer_production.yaml --gpu 1

```

**Parameters:**
- `--length, -l`: Peptide length (5-50 residues, 8-15 recommended for speed)
- `--output, -o`: Output PDB file path (default: auto-generated)
- `--rm_aa`: Amino acids to exclude (default: "C" to avoid disulfides)
- `--add_rg`: Add radius of gyration constraint for compact structures
- `--soft_iters`: Number of optimization iterations (20 for testing, 50+ for production)
- `--quiet`: Suppress verbose output

##### Two Prediction Modes

The script supports two modes:

1. **Hallucination Mode**: Generate both sequence AND structure from scratch
   - Use when you want to design a new cyclic peptide
   - Specify `peptide.length` in config or `--length` on CLI

2. **Sequence Mode**: Predict structure for a GIVEN sequence
   - Use when you have a known sequence and want its 3D structure
   - Specify `peptide.sequence` in config or `--sequence` on CLI
   - Sequence is preserved exactly (no design changes)

#### Design Sequence for Backbone

```bash
# Redesign sequence for existing backbone
ALPHAFOLD_DATA_DIR=./params ./env/bin/python scripts/design_cyclic_sequence.py \                                          
    --input examples/data/test_backbone.pdb \                                                                               
    --chain A \                                                                                                             
    --output results/redesigned.pdb \                                                                                       
    --iterations 100 \                                                                                                      
    --gpu 0 


# Partial redesign (specific positions only)
python scripts/design_cyclic_sequence.py \
  --input examples/data/structures/1P3J.pdb \
  --positions "1-5,10-15" \
  --output results/partial_design.pdb
```

#### Design Binder to Target

```bash
# Design 12-residue binder to target protein
python scripts/design_cyclic_binder.py \
  --target examples/data/structures/1O91.pdb \
  --target_chain A \
  --binder_len 12 \
  --output results/binder.pdb \
  --iterations 100

# Target specific binding site
python scripts/design_cyclic_binder.py \
  --target examples/data/structures/1P3J.pdb \
  --binder_len 14 \
  --hotspot "15-25,30-35" \
  --output results/targeted_binder.pdb
```

---

## MCP Server Installation

### Option 1: Using fastmcp (Recommended)

```bash
# Install MCP server for Claude Code
fastmcp install src/server.py --name cycpep-tools
```

### Option 2: Manual Installation for Claude Code

```bash
# Add MCP server to Claude Code
claude mcp add cycpep-tools -- $(pwd)/env/bin/python $(pwd)/src/server.py

# Verify installation
claude mcp list
```


## Using with Claude Code

After installing the MCP server, you can use it directly in Claude Code.

### Quick Start

```bash
# Start Claude Code
claude
```

### Example Prompts

#### Tool Discovery
```
What tools are available from cycpep-tools?
```

#### Quick Structure Prediction (Sync API)
```
Generate a small 8-residue cyclic peptide structure for testing using predict_cyclic_structure
```

#### Advanced Structure Prediction (Submit API)
```
Submit a high-quality 3D structure prediction job for a 20-residue cyclic peptide excluding cysteines and methionines, with 200 iterations and compact structure constraints
```

#### Sequence Design for Existing Backbone
```
Using @examples/data/test_backbone.pdb, redesign the amino acid sequence for chain A to optimize stability
```

#### Binder Design to Target
```
Design a 12-residue cyclic peptide binder to @examples/data/structures/1O91.pdb chain A, targeting the binding site around residues 15-25
```

#### Job Management
```
Check the status of my submitted job abc12345 and show me the execution logs
```

#### Batch Processing
```
Create a batch job to predict structures for cyclic peptides of lengths 8, 10, 12, and 14 residues, excluding cysteines
```

### Using @ References

In Claude Code, use `@` to reference files and directories:

| Reference | Description |
|-----------|-------------|
| `@examples/data/sequences/sample_cyclic_peptides.txt` | Reference sample sequences |
| `@examples/data/structures/1P3J.pdb` | Reference protein structure |
| `@configs/predict_cyclic_structure_config.json` | Reference config file |
| `@results/` | Reference output directory |

---

### Example Prompts

```bash
# Start Gemini CLI
gemini

# Example prompts (same as Claude Code)
> What tools are available for cyclic peptide analysis?
> Predict structure for 10-residue cyclic peptide with compact constraints
> Check status of job abc12345
```

---

## Available Tools

### Quick Operations (Sync API)

These tools return results immediately (< 10 minutes):

| Tool | Description | Parameters | Use Case |
|------|-------------|------------|----------|
| `predict_cyclic_structure` | Generate 3D structure from scratch | `length`, `rm_aa`, `add_rg`, `soft_iters` | Novel peptide design |
| `design_cyclic_sequence` | Redesign sequence for backbone | `input_file`, `chain`, `positions`, `iterations` | Optimize existing structures |
| `validate_cyclic_peptide_file` | Validate PDB file structure | `file_path` | Quality control |

### Long-Running Tasks (Submit API)

These tools return a job_id for tracking (> 10 minutes):

| Tool | Description | Parameters | Use Case |
|------|-------------|------------|----------|
| `submit_cyclic_binder_design` | Design binders to targets | `target_file`, `binder_len`, `target_chain`, `hotspot` | Drug development |
| `submit_large_structure_prediction` | High-quality structure prediction | `length`, `soft_iters`, `add_rg` | Production structures |
| `submit_large_sequence_design` | High-accuracy sequence design | `input_file`, `iterations`, `positions` | Complex backbone optimization |
| `submit_batch_structure_prediction` | Multiple peptide generation | `lengths`, `rm_aa`, `add_rg` | Virtual library creation |

### Job Management Tools

| Tool | Description | Use Case |
|------|-------------|----------|
| `get_job_status` | Check job progress | Monitor long-running tasks |
| `get_job_result` | Get results when completed | Retrieve final outputs |
| `get_job_log` | View execution logs | Debug issues |
| `cancel_job` | Cancel running job | Stop unwanted tasks |
| `list_jobs` | List all jobs | Job queue management |

---

### Example 2: Production-Quality Structure Prediction

**Goal:** Generate high-quality 3D structure for a medium-sized cyclic peptide

**Using Script:**
```bash
python scripts/predict_cyclic_structure.py \
  --length 15 \
  --output results/production_15mer.pdb \
  --rm_aa "C,M" \
  --add_rg \
  --soft_iters 100
```

**Using MCP (in Claude Code):**
```
Submit a high-quality structure prediction for a 15-residue cyclic peptide excluding cysteines and methionines, with compactness constraints and 100 iterations. Check status and show results when complete.
```

**Expected Process:**
- Submit job → get job_id
- Monitor with `get_job_status`
- Runtime: ~15-25 minutes
- Retrieve results with `get_job_result`
- High-quality structure with pLDDT > 0.85

### Example 3: Sequence Optimization Workflow

**Goal:** Optimize an existing cyclic peptide backbone structure

**Using Script:**
```bash
python scripts/design_cyclic_sequence.py \
  --input examples/data/structures/1P3J.pdb \
  --chain A \
  --positions "1-10" \
  --iterations 150 \
  --output results/optimized_sequence.pdb
```

**Using MCP (in Claude Code):**
```
Using @examples/data/structures/1P3J.pdb, redesign positions 1-10 of chain A to improve the sequence while maintaining the cyclic backbone structure. Use 150 iterations for high accuracy.
```

### Example 4: Drug Discovery Pipeline

**Goal:** Design cyclic peptide binders for drug development

**Using MCP (in Claude Code):**
```
I want to design cyclic peptide binders for drug discovery:

1. Target protein: @examples/data/structures/1O91.pdb chain A
2. Focus on binding site around residues 20-30
3. Test multiple binder lengths: 10, 12, and 14 residues
4. Use high-accuracy settings

Submit jobs for all three lengths and track their progress.
```

**Expected Workflow:**
- Three separate `submit_cyclic_binder_design` jobs
- Monitor progress with job management tools
- Runtime: 20-40 minutes each
- Compare results to select best binder candidates

### Example 5: Virtual Library Generation

**Goal:** Create a library of cyclic peptides for screening

**Using MCP (in Claude Code):**
```
Generate a virtual library of cyclic peptides for screening studies:

Lengths: 8, 10, 12, 14, 16 residues
Exclude: cysteines and methionines
Requirements: compact structures with good drug-like properties
Output: separate PDB files for each peptide

Use batch processing to generate all peptides efficiently.
```

**Expected Output:**
- 5 different cyclic peptides
- All with compact, drug-like properties
- Quality metrics for each structure
- Suitable for further computational screening

---

## Demo Data

The `examples/data/` directory contains sample data for testing:

| File | Description | Use With |
|------|-------------|----------|
| `sequences/sample_cyclic_peptides.txt` | 21 example cyclic peptide sequences | Reference sequences |
| `structures/1P3J.pdb` | Example protein structure | `design_cyclic_sequence`, `validate_cyclic_peptide_file` |
| `structures/1O91.pdb` | Target protein for binder design | `submit_cyclic_binder_design` |
| `structures/test_backbone.pdb` | Sample cyclic backbone | `design_cyclic_sequence` |

### Sample Sequences from `sample_cyclic_peptides.txt`:
- **CycPep_8**: `GGRRWWCG` - Short cyclic peptide with tryptophans
- **CycPep_12**: `GSLDFPWQRLCG` - Medium-sized with mixed properties
- **CycPep_20**: `GALRWPFQNTYSHRCGKIVE` - Large peptide with charged residues

---

## Configuration Files

The `configs/` directory contains configuration templates:

| Config | Description | Key Parameters |
|--------|-------------|----------------|
| `predict_cyclic_structure_config.json` | Structure prediction settings | `soft_iters: 50`, `rm_aa: "C"` |
| `design_cyclic_sequence_config.json` | Sequence design settings | `iterations: 100` |
| `design_cyclic_binder_config.json` | Binder design settings | `interface_cutoff: 8.0` |
| `default_config.json` | Global defaults | Quality thresholds, optimization settings |

### Example Config Usage

```json
{
  "peptide": {
    "length_range": [5, 50],
    "rm_aa": "C"
  },
  "optimization": {
    "soft_iters": 50,
    "stage_iters": [50, 50, 10]
  },
  "quality_thresholds": {
    "plddt_good": 0.70,
    "plddt_excellent": 0.90,
    "pae_excellent": 0.10,
    "pae_good": 0.30
  }
}
```

---

## Troubleshooting

### Environment Issues

**Problem:** Environment not found
```bash
# Recreate environment
$PKG_MGR create -p ./env python=3.10 -y
$PKG_MGR activate ./env
pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install git+https://github.com/sokrypton/ColabDesign.git@v1.1.1
$PKG_MGR install -c conda-forge rdkit -y
pip install fastmcp loguru click tqdm
```

**Problem:** RDKit import errors
```bash
# Install RDKit from conda-forge
$PKG_MGR install -c conda-forge rdkit -y
```

**Problem:** JAX/CUDA issues
```bash
# Verify JAX installation
python -c "import jax; print(f'JAX version: {jax.__version__}, Devices: {jax.devices()}')"

# Reinstall JAX if needed
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**Problem:** Import errors
```bash
# Verify core installations
python -c "
import jax
import numpy as np
import colabdesign
from src.server import mcp
print('All imports successful')
"
```

### MCP Issues

**Problem:** Server not found in Claude Code
```bash
# Check MCP registration
claude mcp list

# Re-add if needed
claude mcp remove cycpep-tools
claude mcp add cycpep-tools -- $(pwd)/env/bin/python $(pwd)/src/server.py
```

**Problem:** Server startup errors
```bash
# Test server directly
$PKG_MGR run -p ./env python src/server.py

# Check dependencies
$PKG_MGR run -p ./env python -c "
from src.server import mcp
print('Server tools:', list(mcp.list_tools().keys())[:5])
"
```

**Problem:** Tools not working
```bash
# Verify tool count
$PKG_MGR run -p ./env python -c "
from src.server import mcp
tools = mcp.list_tools()
print(f'Total tools: {len(tools)}')
print('Sync tools:', [name for name in tools if not name.startswith('submit_')])
print('Submit tools:', [name for name in tools if name.startswith('submit_')])
"
```

### Job Issues

**Problem:** Job stuck in pending
```bash
# Check job directory
ls -la jobs/

# Check job manager
python -c "
import sys; sys.path.append('.')
from src.jobs.manager import job_manager
print(job_manager.list_jobs())
"
```

**Problem:** Job failed with errors
```
Use get_job_log with job_id "your_job_id" and tail 100 to see detailed error messages
```

**Problem:** Out of memory errors
```
- Reduce peptide length (use <15 residues)
- Reduce iterations (use soft_iters=20 for testing)
- Ensure sufficient RAM (4-8GB for typical peptides)
```

### Structure Quality Issues

**Problem:** Low pLDDT scores (<0.5)
```
- Increase soft_iters (try 100+)
- Add compactness constraint (add_rg=True)
- Try different excluded amino acids
- Reduce peptide length for better quality
```

**Problem:** High PAE values (>0.5)
```
- Increase optimization iterations
- Check for problematic amino acid sequences
- Verify cyclic connectivity in output structure
```

### Performance Issues

**Problem:** Very slow computation
```
- Use GPU acceleration if available
- Reduce peptide length for testing
- Use fewer iterations for initial validation
- Check system memory usage
```

**Problem:** AlphaFold parameters not found
```bash
# Parameters should auto-download to params/ directory
ls -la params/
# If missing, restart computation to trigger download
```

---

## Development

### Running Tests

```bash
# Activate environment
$PKG_MGR activate ./env

# Run integration tests
python tests/run_integration_tests.py src/server.py env

# Test individual scripts
python scripts/predict_cyclic_structure.py --length 8 --soft_iters 20 --quiet
```

### Starting Dev Server

```bash
# Run MCP server in development mode
$PKG_MGR run -p ./env fastmcp dev src/server.py

# Test server endpoints
curl http://localhost:6277/health
```

### Performance Monitoring

```bash
# Monitor job queue
python -c "
from src.jobs.manager import job_manager
jobs = job_manager.list_jobs()
print(f'Total jobs: {len(jobs.get(\"jobs\", []))}')
"

# Check system resources
top -p $(pgrep -f "src/server.py")
```

---

## License

Based on ColabDesign framework. See individual component licenses for details.

## Credits

Based on [ColabDesign v1.1.1](https://github.com/sokrypton/ColabDesign) by Sergey Ovchinnikov and team.

**Key Technologies:**
- **ColabDesign**: Protein design framework
- **AlphaFold**: Structure prediction backbone
- **JAX**: High-performance computing
- **FastMCP**: Model Context Protocol server
- **RDKit**: Molecular operations

---

## Quick Reference

### Essential Commands
```bash
# Environment setup
$PKG_MGR activate ./env

# MCP server registration
claude mcp add cycpep-tools -- $(pwd)/env/bin/python $(pwd)/src/server.py

# Quick test
python scripts/predict_cyclic_structure.py --length 8 --soft_iters 20

# Development server
$PKG_MGR run -p ./env fastmcp dev src/server.py
```

### Quality Thresholds
- **pLDDT**: >0.90 (excellent), >0.70 (good), >0.50 (acceptable)
- **PAE**: <0.10 (excellent), <0.30 (good), <0.50 (acceptable)
- **Typical Runtime**: 2-5 min (small), 15-30 min (large/complex)

### Recommended Starting Points
- **Testing**: length=8, soft_iters=20 (~2-3 min)
- **Standard**: length=12, soft_iters=50 (~5-8 min)
- **Production**: length=15+, soft_iters=100+ (submit API)