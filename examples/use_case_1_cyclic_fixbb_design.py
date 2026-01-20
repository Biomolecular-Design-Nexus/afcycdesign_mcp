#!/usr/bin/env python3
"""
Cyclic Peptide Fixed Backbone Design

This script redesigns the amino acid sequence for a given cyclic peptide backbone structure
while maintaining the cyclization constraint (head-to-tail cyclization).

Based on: af_cyc_design.ipynb - cyclic peptide structure prediction and design using AlphaFold
Reference: Stephen Rettie et al., doi: https://doi.org/10.1101/2023.02.25.529956

Usage:
    python use_case_1_cyclic_fixbb_design.py --pdb input.pdb --chain A --output designed_cyclic.pdb
    python use_case_1_cyclic_fixbb_design.py --pdb_code 7m28 --chain A --output designed_cyclic.pdb
"""

import argparse
import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import jax.numpy as jnp
import jax
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.af.alphafold.common import residue_constants


def add_cyclic_offset(model, offset_type=2):
    """
    Add cyclic offset to connect N and C termini for head-to-tail cyclization.

    Args:
        model: AfDesign model instance
        offset_type: Type of offset (1, 2, or 3)
            1 - Basic cyclic offset
            2 - Signed cyclic offset (default)
            3 - Enhanced cyclic offset with scaling
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


def add_rg_loss(model, weight=0.1):
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


def get_pdb_file(pdb_input):
    """
    Get PDB file from various sources.

    Args:
        pdb_input: PDB code (4 characters), file path, or None for upload

    Returns:
        str: Path to PDB file
    """
    if pdb_input is None or pdb_input == "":
        raise ValueError("PDB input is required. Provide --pdb or --pdb_code")
    elif os.path.isfile(pdb_input):
        return pdb_input
    elif len(pdb_input) == 4:
        # Download from PDB
        pdb_file = f"{pdb_input}.pdb"
        if not os.path.exists(pdb_file):
            os.system(f"wget -qnc https://files.rcsb.org/view/{pdb_input}.pdb")
        return pdb_file
    else:
        # Try AlphaFold DB
        af_file = f"AF-{pdb_input}-F1-model_v3.pdb"
        if not os.path.exists(af_file):
            os.system(f"wget -qnc https://alphafold.ebi.ac.uk/files/AF-{pdb_input}-F1-model_v3.pdb")
        return af_file


def design_cyclic_peptide_fixbb(pdb_file, chain="A", offset_type=2, add_rg=False,
                               rg_weight=0.1, output_file="cyclic_fixbb_designed.pdb",
                               num_recycles=0, verbose=True):
    """
    Design a new sequence for a cyclic peptide backbone structure.

    Args:
        pdb_file: Path to input PDB file
        chain: Chain ID to use
        offset_type: Type of cyclic offset (1, 2, or 3)
        add_rg: Whether to add radius of gyration loss
        rg_weight: Weight for RG loss
        output_file: Output PDB file path
        num_recycles: Number of recycles for AlphaFold
        verbose: Whether to print progress

    Returns:
        dict: Results including sequences and metrics
    """
    # Clear previous models
    clear_mem()

    # Initialize model for fixed backbone design
    af_model = mk_afdesign_model(protocol="fixbb", num_recycles=num_recycles)

    if verbose:
        print(f"Loading structure: {pdb_file}, chain: {chain}")

    # Prepare inputs
    af_model.prep_inputs(pdb_filename=pdb_file, chain=chain)

    # Add cyclic offset for head-to-tail cyclization
    add_cyclic_offset(af_model, offset_type=offset_type)

    # Optionally add radius of gyration loss
    if add_rg:
        add_rg_loss(af_model, weight=rg_weight)

    if verbose:
        print(f"Peptide length: {af_model._len}")
        print(f"Loss weights: {af_model.opt['weights']}")

    # Restart and run design
    af_model.restart()

    if verbose:
        print("Running 3-stage design optimization...")

    af_model.design_3stage()

    # Save results
    af_model.save_pdb(output_file)

    if verbose:
        print(f"Design complete! Saved to: {output_file}")
        print(f"Final metrics: {af_model.aux['log']}")

    # Get designed sequences
    sequences = af_model.get_seqs()

    results = {
        "sequences": sequences,
        "pdb_file": output_file,
        "metrics": af_model.aux['log'],
        "length": af_model._len,
        "model": af_model
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Design cyclic peptide sequences for fixed backbone structures"
    )
    parser.add_argument("--pdb", type=str, help="Path to input PDB file")
    parser.add_argument("--pdb_code", type=str, help="4-letter PDB code to download")
    parser.add_argument("--chain", type=str, default="A", help="Chain ID to use (default: A)")
    parser.add_argument("--output", type=str, default="cyclic_fixbb_designed.pdb",
                       help="Output PDB file (default: cyclic_fixbb_designed.pdb)")
    parser.add_argument("--offset_type", type=int, choices=[1, 2, 3], default=2,
                       help="Cyclic offset type (default: 2)")
    parser.add_argument("--add_rg", action="store_true",
                       help="Add radius of gyration loss")
    parser.add_argument("--rg_weight", type=float, default=0.1,
                       help="Weight for RG loss (default: 0.1)")
    parser.add_argument("--num_recycles", type=int, default=0,
                       help="Number of AF2 recycles (default: 0)")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress verbose output")

    args = parser.parse_args()

    # Get PDB file
    pdb_input = args.pdb or args.pdb_code
    if not pdb_input:
        parser.error("Either --pdb or --pdb_code is required")

    try:
        pdb_file = get_pdb_file(pdb_input)

        if not args.quiet:
            print("=== AfCycDesign: Cyclic Peptide Fixed Backbone Design ===")
            print(f"Input: {pdb_file}")
            print(f"Chain: {args.chain}")
            print(f"Output: {args.output}")

        results = design_cyclic_peptide_fixbb(
            pdb_file=pdb_file,
            chain=args.chain,
            offset_type=args.offset_type,
            add_rg=args.add_rg,
            rg_weight=args.rg_weight,
            output_file=args.output,
            num_recycles=args.num_recycles,
            verbose=not args.quiet
        )

        if not args.quiet:
            print("\n=== Results ===")
            for i, seq in enumerate(results["sequences"]):
                print(f"Sequence {i+1}: {seq}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())