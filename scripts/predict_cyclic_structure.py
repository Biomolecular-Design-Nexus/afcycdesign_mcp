#!/usr/bin/env python3
"""
Script: predict_cyclic_structure.py
Description: Predict 3D structure of cyclic peptides from scratch (hallucination)

Original Use Case: examples/use_case_2_cyclic_hallucination.py
Dependencies Removed: None (all essential for core functionality)

Usage:
    python scripts/predict_cyclic_structure.py --length 8 --output structure.pdb
    python scripts/predict_cyclic_structure.py --length 10 --rm_aa "C,M" --add_rg --output compact.pdb

Example:
    python scripts/predict_cyclic_structure.py --length 8 --output examples/data/predicted_8mer.pdb
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
warnings.simplefilter(action='ignore', category=FutureWarning)

# Essential scientific packages for cyclic peptide prediction
import numpy as np
import jax.numpy as jnp
import jax
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.af.alphafold.common import residue_constants

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "rm_aa": "C",
    "offset_type": 2,
    "add_rg": False,
    "rg_weight": 0.1,
    "num_recycles": 0,
    "soft_iters": 50,
    "stage_iters": [50, 50, 10],
    "contact_cutoff": 21.6875,
    "loss_weights": {
        "pae": 1,
        "plddt": 1,
        "con": 0.5
    }
}

# ==============================================================================
# Core Functions (extracted and simplified from use case)
# ==============================================================================
def add_cyclic_offset(model: Any, offset_type: int = 2) -> None:
    """
    Add cyclic offset to connect N and C termini for head-to-tail cyclization.

    Extracted from: examples/use_case_2_cyclic_hallucination.py:30-67

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

    Extracted from: examples/use_case_2_cyclic_hallucination.py:70-87

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


def validate_inputs(length: int, rm_aa: str) -> None:
    """Validate input parameters for cyclic peptide prediction."""
    if length < 5:
        raise ValueError("Peptide length must be at least 5 residues")
    if length > 50:
        print("Warning: Very long peptides (>50) may be challenging to design", file=sys.stderr)

    # Validate amino acid exclusions
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    if rm_aa:
        excluded = set(rm_aa.replace(",", "").replace(" ", "").upper())
        invalid = excluded - valid_aa
        if invalid:
            raise ValueError(f"Invalid amino acids in rm_aa: {invalid}")


def save_output(pdb_file: str, sequences: List[str], metrics: Dict[str, Any],
                metadata: Dict[str, Any]) -> None:
    """Save prediction results and metadata."""
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
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_predict_cyclic_structure(
    length: int,
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Predict/hallucinate a cyclic peptide 3D structure from scratch.

    Args:
        length: Length of peptide to generate (5-50 residues)
        output_file: Path to save PDB file (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - result: Predicted sequences and structure data
            - output_file: Path to output PDB file (if saved)
            - metadata: Execution metadata
            - metrics: Quality metrics (pLDDT, PAE, contacts)

    Example:
        >>> result = run_predict_cyclic_structure(8, "output.pdb")
        >>> print(f"Sequence: {result['result']['sequences'][0]}")
        >>> print(f"pLDDT: {result['metrics']['plddt']:.3f}")
    """
    # Setup configuration
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Validate inputs
    validate_inputs(length, config.get("rm_aa", "C"))

    # Clear any previous models
    clear_mem()

    # Initialize AlphaFold model for hallucination
    af_model = mk_afdesign_model(
        protocol="hallucination",
        num_recycles=config.get("num_recycles", 0)
    )

    # Prepare inputs
    af_model.prep_inputs(length=length, rm_aa=config.get("rm_aa", "C"))

    # Add cyclic offset for head-to-tail cyclization
    add_cyclic_offset(af_model, offset_type=config.get("offset_type", 2))

    # Optionally add radius of gyration loss for compact structures
    if config.get("add_rg", False):
        add_rg_loss(af_model, weight=config.get("rg_weight", 0.1))

    # Pre-design with Gumbel initialization and softmax activation
    af_model.restart()
    af_model.set_seq(mode="gumbel")

    # Configure contact loss
    af_model.set_opt("con", binary=True,
                    cutoff=config.get("contact_cutoff", 21.6875),
                    num=af_model._len, seqsep=0)

    # Set loss weights
    weights = config.get("loss_weights", DEFAULT_CONFIG["loss_weights"])
    af_model.set_weights(**weights)

    # Run soft optimization
    soft_iters = config.get("soft_iters", 50)
    af_model.design_soft(soft_iters)

    # Three-stage design: logits → soft → hard
    stage_iters = config.get("stage_iters", [50, 50, 10])
    af_model.set_seq(seq=af_model.aux["seq"]["pseudo"])
    af_model.design_3stage(*stage_iters)

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
            "length": af_model._len,
            "config": config,
            "protocol": "hallucination"
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
            "length": af_model._len,
            "config": config,
            "protocol": "hallucination"
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
    parser.add_argument('--length', '-l', type=int, required=True,
                       help='Length of peptide to generate (5-50 residues)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output PDB file path')
    parser.add_argument('--config', '-c', type=str,
                       help='Config file (JSON)')

    # Core parameters
    parser.add_argument('--rm_aa', type=str, default="C",
                       help='Amino acids to exclude (comma-separated, default: C)')
    parser.add_argument('--offset_type', type=int, choices=[1, 2, 3], default=2,
                       help='Cyclic offset type (default: 2)')
    parser.add_argument('--add_rg', action="store_true",
                       help='Add radius of gyration loss for compact structures')
    parser.add_argument('--rg_weight', type=float, default=0.1,
                       help='Weight for RG loss (default: 0.1)')

    # Optimization parameters
    parser.add_argument('--num_recycles', type=int, default=0,
                       help='Number of AF2 recycles (default: 0)')
    parser.add_argument('--soft_iters', type=int, default=50,
                       help='Iterations for soft pre-design (default: 50)')
    parser.add_argument('--stage_iters', type=int, nargs=3, default=[50, 50, 10],
                       help='Iterations for 3-stage design: logits soft hard (default: 50 50 10)')

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
    if args.rm_aa != "C":
        overrides["rm_aa"] = args.rm_aa
    if args.offset_type != 2:
        overrides["offset_type"] = args.offset_type
    if args.add_rg:
        overrides["add_rg"] = True
    if args.rg_weight != 0.1:
        overrides["rg_weight"] = args.rg_weight
    if args.num_recycles != 0:
        overrides["num_recycles"] = args.num_recycles
    if args.soft_iters != 50:
        overrides["soft_iters"] = args.soft_iters
    if args.stage_iters != [50, 50, 10]:
        overrides["stage_iters"] = args.stage_iters

    try:
        if not args.quiet:
            print("=== Cyclic Peptide Structure Prediction ===")
            print(f"Length: {args.length}")
            print(f"Excluded AAs: {args.rm_aa}")
            if args.output:
                print(f"Output: {args.output}")
            if args.add_rg:
                print(f"Radius of gyration constraint: weight={args.rg_weight}")

        # Run prediction
        result = run_predict_cyclic_structure(
            length=args.length,
            output_file=args.output,
            config=config,
            **overrides
        )

        if not args.quiet:
            print("\n=== Results ===")
            for i, seq in enumerate(result["result"]["sequences"]):
                print(f"Sequence {i+1}: {seq}")

            # Display metrics
            metrics = result["metrics"]
            if "plddt" in metrics:
                print(f"pLDDT: {metrics['plddt']:.3f}")
            if "pae" in metrics:
                print(f"PAE: {metrics['pae']:.3f}")
            if "con" in metrics:
                print(f"Contacts: {metrics['con']:.3f}")

            if result["output_file"]:
                print(f"\nPDB saved to: {result['output_file']}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())