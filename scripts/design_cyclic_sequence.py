#!/usr/bin/env python3
"""
Script: design_cyclic_sequence.py
Description: Redesign amino acid sequences for given cyclic peptide backbone structures

Original Use Case: examples/use_case_1_cyclic_fixbb_design.py
Dependencies Removed: PDB downloading logic simplified, file validation inlined

Usage:
    python scripts/design_cyclic_sequence.py --input backbone.pdb --chain A --output designed.pdb
    python scripts/design_cyclic_sequence.py --input backbone.pdb --chain A --positions "1-5,10" --output specific_design.pdb

Example:
    python scripts/design_cyclic_sequence.py --input examples/data/structures/peptide.pdb --chain A --output results/redesigned.pdb
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import json
import sys
import warnings
import os
warnings.simplefilter(action='ignore', category=FutureWarning)

# Essential scientific packages for sequence design
import numpy as np
import jax.numpy as jnp
import jax
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.af.alphafold.common import residue_constants

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "chain": "A",
    "offset_type": 2,
    "add_rg": False,
    "rg_weight": 0.1,
    "num_recycles": 0,
    "iterations": 100,
    "loss_weights": {
        "dgram_cce": 1.0,
        "fape": 1.0,
        "plddt": 1.0,
        "pae": 0.01
    },
    "positions": None  # Design all positions by default
}

# ==============================================================================
# Utility Functions (simplified from repo)
# ==============================================================================
def validate_pdb_file(pdb_file: Union[str, Path]) -> Path:
    """Validate PDB file exists and is readable."""
    pdb_path = Path(pdb_file)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")
    if not pdb_path.is_file():
        raise ValueError(f"Not a file: {pdb_path}")
    if pdb_path.stat().st_size == 0:
        raise ValueError(f"Empty PDB file: {pdb_path}")
    return pdb_path


def parse_positions(positions_str: Optional[str]) -> Optional[List[int]]:
    """
    Parse position specification string to list of residue numbers.

    Args:
        positions_str: String like "1-5,10,15-20" or None for all positions

    Returns:
        List of position indices (0-based) or None for all positions
    """
    if not positions_str:
        return None

    positions = []
    for part in positions_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            positions.extend(range(start-1, end))  # Convert to 0-based
        else:
            positions.append(int(part) - 1)  # Convert to 0-based

    return sorted(set(positions))


def validate_chain(pdb_file: Path, chain: str) -> None:
    """Validate that the specified chain exists in the PDB file."""
    chains_found = set()
    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    chains_found.add(line[21])
    except Exception as e:
        raise ValueError(f"Error reading PDB file: {e}")

    if chain not in chains_found:
        raise ValueError(f"Chain '{chain}' not found in PDB. Available chains: {sorted(chains_found)}")


def save_output(pdb_file: str, sequences: List[str], metrics: Dict[str, Any],
                metadata: Dict[str, Any]) -> None:
    """Save design results and metadata."""
    output_path = Path(pdb_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Metadata file alongside PDB
    metadata_file = output_path.with_suffix('.json')
    result_data = {
        "sequences": sequences,
        "metrics": metrics,
        "metadata": metadata,
        "pdb_file": str(output_path)
    }

    with open(metadata_file, 'w') as f:
        json.dump(result_data, f, indent=2, default=str)


# ==============================================================================
# Core Functions (extracted and simplified from use case)
# ==============================================================================
def add_cyclic_offset(model: Any, offset_type: int = 2) -> None:
    """
    Add cyclic offset to connect N and C termini for head-to-tail cyclization.

    Extracted from: examples/use_case_1_cyclic_fixbb_design.py:29-69

    Args:
        model: AfDesign model instance
        offset_type: Type of offset (1, 2, or 3)
    """
    def cyclic_offset(L):
        i = np.arange(L)
        ij = np.stack([i, i+L], -1)
        offset = i[:,None] - i[None,:]
        c_offset = np.abs(ij[:,None,:,None] - ij[None,:,None,:]).min((2,3))

        if offset_type == 1:
            c_offset = c_offset
        elif offset_type >= 2:
            a = c_offset < np.abs(offset)
            c_offset[a] = -c_offset[a]
        if offset_type == 3:
            idx = np.abs(c_offset) > 2
            c_offset[idx] = (32 * c_offset[idx]) / abs(c_offset[idx])
        return c_offset * np.sign(offset)

    idx = model._inputs["residue_index"]
    offset = np.array(idx[:,None] - idx[None,:])

    if model.protocol == "binder":
        c_offset = cyclic_offset(model._binder_len)
        offset[model._target_len:, model._target_len:] = c_offset

    if model.protocol in ["fixbb", "partial", "hallucination"]:
        Ln = 0
        for L in model._lengths:
            offset[Ln:Ln+L, Ln:Ln+L] = cyclic_offset(L)
            Ln += L

    model._inputs["offset"] = offset


def add_rg_loss(model: Any, weight: float = 0.1) -> None:
    """
    Add radius of gyration loss to maintain compact structure.

    Args:
        model: AfDesign model instance
        weight: Weight for the RG loss term
    """
    def loss_fn(inputs, outputs):
        xyz = outputs["structure_module"]
        ca = xyz["final_atom_positions"][:,residue_constants.atom_order["CA"]]
        rg = jnp.sqrt(jnp.square(ca - ca.mean(0)).sum(-1).mean() + 1e-8)
        rg_th = 2.38 * ca.shape[0] ** 0.365
        rg = jax.nn.elu(rg - rg_th)
        return {"rg": rg}

    model._callbacks["model"]["loss"].append(loss_fn)
    model.opt["weights"]["rg"] = weight


# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_design_cyclic_sequence(
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Design amino acid sequence for a given cyclic peptide backbone structure.

    Args:
        input_file: Path to input PDB file with backbone structure
        output_file: Path to save designed PDB file (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - result: Designed sequences and structure data
            - output_file: Path to output PDB file (if saved)
            - metadata: Execution metadata
            - metrics: Design quality metrics

    Example:
        >>> result = run_design_cyclic_sequence("input.pdb", "output.pdb")
        >>> print(f"Designed sequence: {result['result']['sequences'][0]}")
        >>> print(f"pLDDT: {result['metrics']['plddt']:.3f}")
    """
    # Setup configuration
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Validate inputs
    input_path = validate_pdb_file(input_file)
    chain = config.get("chain", "A")
    validate_chain(input_path, chain)

    # Parse positions if specified
    positions = parse_positions(config.get("positions"))

    # Clear any previous models
    clear_mem()

    # Initialize AlphaFold model for fixed backbone design
    af_model = mk_afdesign_model(
        protocol="fixbb",
        num_recycles=config.get("num_recycles", 0)
    )

    # Prepare inputs from PDB
    af_model.prep_inputs(pdb_filename=str(input_path), chain=chain)

    # Add cyclic offset for head-to-tail cyclization
    add_cyclic_offset(af_model, offset_type=config.get("offset_type", 2))

    # Optionally add radius of gyration loss
    if config.get("add_rg", False):
        add_rg_loss(af_model, weight=config.get("rg_weight", 0.1))

    # Set design positions (all by default)
    if positions is not None:
        # Restrict design to specific positions
        af_model.restart(opt=False)
        af_model._inputs["rm_aa"] = []
        # ColabDesign uses pos notation for fixed positions
        all_pos = set(range(af_model._len))
        fixed_pos = all_pos - set(positions)
        if fixed_pos:
            af_model.set_opt("fix_pos", list(fixed_pos))
    else:
        # Design all positions
        af_model.restart()

    # Set loss weights
    weights = config.get("loss_weights", DEFAULT_CONFIG["loss_weights"])
    af_model.set_weights(**weights)

    # Run design optimization
    iterations = config.get("iterations", 100)
    af_model.design_logits(iterations)

    # Get results
    sequences = af_model.get_seqs()
    metrics = af_model.aux.get('log', {})

    # Save output if requested
    output_path = None
    if output_file:
        output_path = Path(output_file)
        af_model.save_pdb(str(output_path))

        # Save metadata
        metadata = {
            "input_file": str(input_path),
            "chain": chain,
            "length": af_model._len,
            "positions_designed": positions,
            "config": config,
            "protocol": "fixbb"
        }
        save_output(str(output_path), sequences, metrics, metadata)

    return {
        "result": {
            "sequences": sequences,
            "model": af_model
        },
        "output_file": str(output_path) if output_path else None,
        "metrics": metrics,
        "metadata": {
            "input_file": str(input_path),
            "chain": chain,
            "length": af_model._len,
            "positions_designed": positions,
            "config": config,
            "protocol": "fixbb"
        }
    }


# ==============================================================================
# CLI Interface
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input PDB file with backbone structure')
    parser.add_argument('--output', '-o', type=str,
                       help='Output PDB file path for designed sequence')
    parser.add_argument('--config', '-c', type=str,
                       help='Config file (JSON)')

    # Core parameters
    parser.add_argument('--chain', type=str, default="A",
                       help='Chain ID to design (default: A)')
    parser.add_argument('--positions', type=str,
                       help='Positions to design (e.g., "1-5,10,15-20", default: all)')
    parser.add_argument('--offset_type', type=int, choices=[1, 2, 3], default=2,
                       help='Cyclic offset type (default: 2)')

    # Optional constraints
    parser.add_argument('--add_rg', action="store_true",
                       help='Add radius of gyration loss for compact structures')
    parser.add_argument('--rg_weight', type=float, default=0.1,
                       help='Weight for RG loss (default: 0.1)')

    # Optimization parameters
    parser.add_argument('--num_recycles', type=int, default=0,
                       help='Number of AF2 recycles (default: 0)')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Design iterations (default: 100)')

    parser.add_argument('--quiet', action="store_true",
                       help='Suppress verbose output')

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Prepare config overrides from CLI args
    overrides = {}
    if args.chain != "A":
        overrides["chain"] = args.chain
    if args.positions:
        overrides["positions"] = args.positions
    if args.offset_type != 2:
        overrides["offset_type"] = args.offset_type
    if args.add_rg:
        overrides["add_rg"] = True
    if args.rg_weight != 0.1:
        overrides["rg_weight"] = args.rg_weight
    if args.num_recycles != 0:
        overrides["num_recycles"] = args.num_recycles
    if args.iterations != 100:
        overrides["iterations"] = args.iterations

    try:
        if not args.quiet:
            print("=== Cyclic Peptide Sequence Design ===")
            print(f"Input: {args.input}")
            print(f"Chain: {args.chain}")
            if args.positions:
                print(f"Design positions: {args.positions}")
            else:
                print("Design positions: all")
            if args.output:
                print(f"Output: {args.output}")

        # Run design
        result = run_design_cyclic_sequence(
            input_file=args.input,
            output_file=args.output,
            config=config,
            **overrides
        )

        if not args.quiet:
            print("\n=== Results ===")
            for i, seq in enumerate(result["result"]["sequences"]):
                print(f"Designed sequence {i+1}: {seq}")

            # Display metrics
            metrics = result["metrics"]
            if "plddt" in metrics:
                print(f"pLDDT: {metrics['plddt']:.3f}")
            if "pae" in metrics:
                print(f"PAE: {metrics['pae']:.3f}")

            if result["output_file"]:
                print(f"\nDesigned structure saved to: {result['output_file']}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())