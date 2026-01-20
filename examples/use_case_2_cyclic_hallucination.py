#!/usr/bin/env python3
"""
Cyclic Peptide Hallucination

This script generates/hallucinates cyclic peptides from scratch for a given length.
AlphaFold predicts well-structured proteins (high pLDDT, low PAE, many contacts) with
head-to-tail cyclization constraints.

Based on: af_cyc_design.ipynb - cyclic peptide structure prediction and design using AlphaFold
Reference: Stephen Rettie et al., doi: https://doi.org/10.1101/2023.02.25.529956

Usage:
    python use_case_2_cyclic_hallucination.py --length 13 --output hallucinated_cyclic.pdb
    python use_case_2_cyclic_hallucination.py --length 15 --rm_aa "C,M" --add_rg --output compact_cyclic.pdb
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


def hallucinate_cyclic_peptide(length=13, rm_aa="C", add_rg=False, rg_weight=0.1,
                              offset_type=2, output_file="cyclic_hallucinated.pdb",
                              num_recycles=0, soft_iters=50, stage_iters=(50, 50, 10),
                              verbose=True):
    """
    Hallucinate a cyclic peptide structure from scratch.

    Args:
        length: Length of peptide to generate
        rm_aa: Amino acids to remove (e.g., "C" or "C,M")
        add_rg: Whether to add radius of gyration loss
        rg_weight: Weight for RG loss
        offset_type: Type of cyclic offset
        output_file: Output PDB file path
        num_recycles: Number of recycles for AlphaFold
        soft_iters: Iterations for soft pre-design
        stage_iters: Iterations for 3-stage design (logits, soft, hard)
        verbose: Whether to print progress

    Returns:
        dict: Results including sequences and metrics
    """
    # Clear previous models
    clear_mem()

    # Initialize model for hallucination
    af_model = mk_afdesign_model(protocol="hallucination", num_recycles=num_recycles)

    if verbose:
        print(f"Initializing cyclic peptide hallucination (length: {length})")

    # Prepare inputs
    af_model.prep_inputs(length=length, rm_aa=rm_aa)

    # Add cyclic offset for head-to-tail cyclization
    add_cyclic_offset(af_model, offset_type=offset_type)

    # Optionally add radius of gyration loss
    if add_rg:
        add_rg_loss(af_model, weight=rg_weight)

    if verbose:
        print(f"Peptide length: {af_model._len}")
        print(f"Loss weights: {af_model.opt['weights']}")
        print(f"Removed amino acids: {rm_aa}")

    # Pre-design with Gumbel initialization and softmax activation
    if verbose:
        print("Starting pre-design with Gumbel initialization...")

    af_model.restart()
    af_model.set_seq(mode="gumbel")

    # Configure contact loss
    af_model.set_opt("con", binary=True, cutoff=21.6875, num=af_model._len, seqsep=0)
    af_model.set_weights(pae=1, plddt=1, con=0.5)

    # Run soft optimization
    af_model.design_soft(soft_iters)

    if verbose:
        print("Running 3-stage design optimization...")

    # Three-stage design: logits → soft → hard
    af_model.set_seq(seq=af_model.aux["seq"]["pseudo"])
    af_model.design_3stage(*stage_iters)

    # Save results
    af_model.save_pdb(output_file)

    if verbose:
        print(f"Hallucination complete! Saved to: {output_file}")
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
        description="Hallucinate cyclic peptide structures from scratch"
    )
    parser.add_argument("--length", type=int, required=True,
                       help="Length of peptide to generate")
    parser.add_argument("--rm_aa", type=str, default="C",
                       help="Amino acids to exclude (comma-separated, default: C)")
    parser.add_argument("--output", type=str, default="cyclic_hallucinated.pdb",
                       help="Output PDB file (default: cyclic_hallucinated.pdb)")
    parser.add_argument("--offset_type", type=int, choices=[1, 2, 3], default=2,
                       help="Cyclic offset type (default: 2)")
    parser.add_argument("--add_rg", action="store_true",
                       help="Add radius of gyration loss for compact structures")
    parser.add_argument("--rg_weight", type=float, default=0.1,
                       help="Weight for RG loss (default: 0.1)")
    parser.add_argument("--num_recycles", type=int, default=0,
                       help="Number of AF2 recycles (default: 0)")
    parser.add_argument("--soft_iters", type=int, default=50,
                       help="Iterations for soft pre-design (default: 50)")
    parser.add_argument("--stage_iters", type=int, nargs=3, default=[50, 50, 10],
                       help="Iterations for 3-stage design: logits soft hard (default: 50 50 10)")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress verbose output")

    args = parser.parse_args()

    # Validate inputs
    if args.length < 5:
        parser.error("Peptide length must be at least 5 residues")
    if args.length > 50:
        print("Warning: Very long peptides (>50) may be challenging to design", file=sys.stderr)

    try:
        if not args.quiet:
            print("=== AfCycDesign: Cyclic Peptide Hallucination ===")
            print(f"Length: {args.length}")
            print(f"Excluded AAs: {args.rm_aa}")
            print(f"Output: {args.output}")
            if args.add_rg:
                print(f"Radius of gyration constraint: weight={args.rg_weight}")

        results = hallucinate_cyclic_peptide(
            length=args.length,
            rm_aa=args.rm_aa,
            add_rg=args.add_rg,
            rg_weight=args.rg_weight,
            offset_type=args.offset_type,
            output_file=args.output,
            num_recycles=args.num_recycles,
            soft_iters=args.soft_iters,
            stage_iters=args.stage_iters,
            verbose=not args.quiet
        )

        if not args.quiet:
            print("\n=== Results ===")
            for i, seq in enumerate(results["sequences"]):
                print(f"Sequence {i+1}: {seq}")

            # Additional metrics
            metrics = results["metrics"]
            if "plddt" in metrics:
                print(f"pLDDT: {metrics['plddt']:.3f}")
            if "pae" in metrics:
                print(f"PAE: {metrics['pae']:.3f}")
            if "con" in metrics:
                print(f"Contacts: {metrics['con']:.3f}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())