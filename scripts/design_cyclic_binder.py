#!/usr/bin/env python3
"""
Script: design_cyclic_binder.py
Description: Design cyclic peptide binders that bind to target protein structures

Original Use Case: examples/use_case_3_cyclic_binder_design.py
Dependencies Removed: PDB downloading simplified, scipy imports minimized

Usage:
    python scripts/design_cyclic_binder.py --target protein.pdb --target_chain A --binder_len 10 --output binder.pdb
    python scripts/design_cyclic_binder.py --target protein.pdb --target_chain A --binder_len 12 --hotspot "1-10,15" --output specific_binder.pdb

Example:
    python scripts/design_cyclic_binder.py --target examples/data/structures/target.pdb --target_chain A --binder_len 8 --output results/cyclic_binder.pdb
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
import re
warnings.simplefilter(action='ignore', category=FutureWarning)

# Essential scientific packages for binder design
import numpy as np
import jax.numpy as jnp
import jax
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.af.alphafold.common import residue_constants
from colabdesign.shared.utils import copy_dict

# Optional: scipy for softmax (inline fallback provided)
try:
    from scipy.special import softmax
except ImportError:
    def softmax(x, axis=-1):
        """Simple softmax implementation as fallback."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "target_chain": "A",
    "binder_len": 10,
    "offset_type": 2,
    "num_recycles": 0,
    "iterations": 100,
    "hotspot": None,  # All positions by default
    "loss_weights": {
        "dgram_cce": 1.0,
        "fape": 1.0,
        "plddt": 1.0,
        "pae": 0.01,
        "i_pae": 1.0,
        "i_con": 1.0
    },
    "interface_cutoff": 8.0,  # Angstroms for interface contacts
    "n_grad_steps": 100,
    "grad_check": 10
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


def parse_hotspot(hotspot_str: Optional[str]) -> Optional[List[int]]:
    """
    Parse hotspot specification string to list of residue numbers.

    Args:
        hotspot_str: String like "1-5,10,15-20" or None for no hotspot

    Returns:
        List of residue indices (0-based) or None for no hotspot
    """
    if not hotspot_str:
        return None

    residues = []
    for part in hotspot_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = map(int, part.split('-'))
            residues.extend(range(start-1, end))  # Convert to 0-based
        else:
            residues.append(int(part) - 1)  # Convert to 0-based

    return sorted(set(residues))


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
    """Save binder design results and metadata."""
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

    Extracted from: examples/use_case_3_cyclic_binder_design.py:33-70

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


def parse_pdb_input(pdb_input: str) -> str:
    """
    Get PDB file from various sources (simplified version).

    Args:
        pdb_input: PDB file path, PDB code, or UniProt ID

    Returns:
        str: Path to PDB file
    """
    if os.path.isfile(pdb_input):
        return pdb_input
    elif len(pdb_input) == 4:
        # Assume PDB code - user should download manually or provide file
        raise ValueError(f"PDB code '{pdb_input}' provided but file not found. "
                        f"Please download {pdb_input}.pdb manually or provide file path.")
    else:
        # Assume it's a file path that doesn't exist
        raise FileNotFoundError(f"PDB file not found: {pdb_input}")


# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_design_cyclic_binder(
    target_file: Union[str, Path],
    binder_len: int,
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Design a cyclic peptide binder that binds to a target protein structure.

    Args:
        target_file: Path to target protein PDB file
        binder_len: Length of binder peptide to design (6-20 residues recommended)
        output_file: Path to save designed complex PDB file (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - result: Designed binder sequences and complex structure
            - output_file: Path to output PDB file (if saved)
            - metadata: Execution metadata
            - metrics: Design quality metrics (interface contacts, pLDDT)

    Example:
        >>> result = run_design_cyclic_binder("target.pdb", 10, "binder_complex.pdb")
        >>> print(f"Binder sequence: {result['result']['sequences'][0]}")
        >>> print(f"Interface pLDDT: {result['metrics']['i_plddt']:.3f}")
    """
    # Setup configuration
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Validate inputs
    target_path = validate_pdb_file(target_file)
    target_chain = config.get("target_chain", "A")
    validate_chain(target_path, target_chain)

    if binder_len < 6:
        print("Warning: Very short binders (<6 residues) may be unstable", file=sys.stderr)
    if binder_len > 20:
        print("Warning: Long binders (>20 residues) may be challenging to design", file=sys.stderr)

    # Parse hotspot if specified
    hotspot = parse_hotspot(config.get("hotspot"))

    # Clear any previous models
    clear_mem()

    # Initialize AlphaFold model for binder design
    af_model = mk_afdesign_model(
        protocol="binder",
        num_recycles=config.get("num_recycles", 0)
    )

    # Prepare inputs from target PDB
    af_model.prep_inputs(
        pdb_filename=str(target_path),
        chain=target_chain,
        binder_len=binder_len
    )

    # Add cyclic offset for head-to-tail cyclization of the binder
    add_cyclic_offset(af_model, offset_type=config.get("offset_type", 2))

    # Set loss weights
    weights = config.get("loss_weights", DEFAULT_CONFIG["loss_weights"])
    af_model.set_weights(**weights)

    # Configure interface constraints
    interface_cutoff = config.get("interface_cutoff", 8.0)
    af_model.set_opt("i_con", cutoff=interface_cutoff)

    # Add hotspot constraints if specified
    if hotspot is not None:
        af_model.set_opt("hotspot", hotspot)

    # Initialize design
    af_model.restart()

    # Run design optimization
    iterations = config.get("iterations", 100)
    grad_steps = config.get("n_grad_steps", 100)
    grad_check = config.get("grad_check", 10)

    # Design the binder sequence
    af_model.design_logits(iterations, temp=1.0)

    # Additional gradient-based optimization
    for i in range(grad_steps):
        af_model.design_logits(1, temp=1.0)
        if i % grad_check == 0:
            # Check convergence
            current_loss = af_model.aux["log"].get("loss", float('inf'))
            if current_loss < 0.1:  # Convergence threshold
                break

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
            "target_file": str(target_path),
            "target_chain": target_chain,
            "binder_length": binder_len,
            "hotspot": hotspot,
            "config": config,
            "protocol": "binder"
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
            "target_file": str(target_path),
            "target_chain": target_chain,
            "binder_length": binder_len,
            "hotspot": hotspot,
            "config": config,
            "protocol": "binder"
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
    parser.add_argument('--target', '-t', type=str, required=True,
                       help='Target protein PDB file')
    parser.add_argument('--target_chain', type=str, default="A",
                       help='Target chain ID (default: A)')
    parser.add_argument('--binder_len', '-l', type=int, required=True,
                       help='Binder peptide length (6-20 recommended)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output PDB file for designed binder complex')
    parser.add_argument('--config', '-c', type=str,
                       help='Config file (JSON)')

    # Core parameters
    parser.add_argument('--hotspot', type=str,
                       help='Target hotspot residues (e.g., "1-5,10,15-20")')
    parser.add_argument('--offset_type', type=int, choices=[1, 2, 3], default=2,
                       help='Cyclic offset type (default: 2)')

    # Optimization parameters
    parser.add_argument('--num_recycles', type=int, default=0,
                       help='Number of AF2 recycles (default: 0)')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Design iterations (default: 100)')
    parser.add_argument('--interface_cutoff', type=float, default=8.0,
                       help='Interface contact cutoff in Angstroms (default: 8.0)')

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
    if args.target_chain != "A":
        overrides["target_chain"] = args.target_chain
    if args.hotspot:
        overrides["hotspot"] = args.hotspot
    if args.offset_type != 2:
        overrides["offset_type"] = args.offset_type
    if args.num_recycles != 0:
        overrides["num_recycles"] = args.num_recycles
    if args.iterations != 100:
        overrides["iterations"] = args.iterations
    if args.interface_cutoff != 8.0:
        overrides["interface_cutoff"] = args.interface_cutoff

    try:
        if not args.quiet:
            print("=== Cyclic Peptide Binder Design ===")
            print(f"Target: {args.target}")
            print(f"Target chain: {args.target_chain}")
            print(f"Binder length: {args.binder_len}")
            if args.hotspot:
                print(f"Hotspot residues: {args.hotspot}")
            if args.output:
                print(f"Output: {args.output}")

        # Run binder design
        result = run_design_cyclic_binder(
            target_file=args.target,
            binder_len=args.binder_len,
            output_file=args.output,
            config=config,
            **overrides
        )

        if not args.quiet:
            print("\n=== Results ===")
            for i, seq in enumerate(result["result"]["sequences"]):
                print(f"Binder sequence {i+1}: {seq}")

            # Display metrics
            metrics = result["metrics"]
            if "i_plddt" in metrics:
                print(f"Interface pLDDT: {metrics['i_plddt']:.3f}")
            if "i_pae" in metrics:
                print(f"Interface PAE: {metrics['i_pae']:.3f}")
            if "i_con" in metrics:
                print(f"Interface contacts: {metrics['i_con']:.3f}")

            if result["output_file"]:
                print(f"\nBinder complex saved to: {result['output_file']}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())