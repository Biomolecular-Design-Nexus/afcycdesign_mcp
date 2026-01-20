#!/usr/bin/env python3
"""
Cyclic Peptide Binder Design

This script designs cyclic peptide binders that bind to a target protein structure.
The designed peptide will have head-to-tail cyclization while maximizing contacts
at the interface and pLDDT of the binder.

Based on: peptide_binder_design.ipynb with cyclic modifications
Reference: ColabDesign peptide binder hallucination workflow

Usage:
    python use_case_3_cyclic_binder_design.py --pdb 4N5T --target_chain A --binder_len 14 --output cyclic_binder.pdb
    python use_case_3_cyclic_binder_design.py --pdb input.pdb --target_chain A --binder_len 12 --hotspot "1-10,15" --output specific_binder.pdb
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
from colabdesign.shared.utils import copy_dict
from scipy.special import softmax
import re


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


def design_cyclic_peptide_binder(pdb_file, target_chain="A", binder_len=14,
                                binder_seq=None, target_hotspot=None,
                                target_flexible=False, use_multimer=False,
                                optimizer="pssm_semigreedy", num_recycles=0,
                                num_models=2, output_file="cyclic_binder.pdb",
                                verbose=True):
    """
    Design a cyclic peptide binder for a target protein structure.

    Args:
        pdb_file: Path to target protein PDB file
        target_chain: Chain ID of target protein
        binder_len: Length of binder peptide to design
        binder_seq: Initial binder sequence (optional)
        target_hotspot: Restrict binding to specific positions (e.g., "1-10,12,15")
        target_flexible: Allow target backbone flexibility
        use_multimer: Use AlphaFold-multimer
        optimizer: Optimization method
        num_recycles: Number of AF2 recycles
        num_models: Number of models to use
        output_file: Output PDB file path
        verbose: Whether to print progress

    Returns:
        dict: Results including sequences and metrics
    """
    # Clear previous models
    clear_mem()

    # Process binder sequence if provided
    if binder_seq:
        binder_seq = re.sub("[^A-Z]", "", binder_seq.upper())
        if len(binder_seq) > 0:
            binder_len = len(binder_seq)
        else:
            binder_seq = None

    # Prepare inputs
    x = {
        "pdb_filename": pdb_file,
        "chain": target_chain,
        "binder_len": binder_len,
        "hotspot": target_hotspot,
        "use_multimer": use_multimer,
        "rm_target_seq": target_flexible
    }

    # Initialize model for binder design
    model = mk_afdesign_model(
        protocol="binder",
        use_multimer=use_multimer,
        num_recycles=num_recycles,
        recycle_mode="sample"
    )

    model.prep_inputs(**x, ignore_missing=False)

    # Add cyclic constraint for binder
    add_cyclic_offset(model, offset_type=2)

    if verbose:
        print(f"Target length: {model._target_len}")
        print(f"Binder length: {model._binder_len}")
        print(f"Hotspot: {target_hotspot or 'None (full interface)'}")

    # Set optimizer and models
    models = model._model_names[:num_models]
    flags = {
        "num_recycles": num_recycles,
        "models": models,
        "dropout": True
    }

    # Restart model with initial sequence
    model.restart(seq=binder_seq)

    if verbose:
        print(f"Running {optimizer} optimization...")

    # Run optimization based on method
    if optimizer == "3stage":
        model.design_3stage(120, 60, 10, **flags)
        pssm = softmax(model._tmp["seq_logits"], -1)

    elif optimizer == "pssm_semigreedy":
        model.design_pssm_semigreedy(120, 32, **flags)
        pssm = softmax(model._tmp["seq_logits"], 1)

    elif optimizer == "semigreedy":
        model.design_pssm_semigreedy(0, 32, **flags)
        pssm = None

    elif optimizer == "pssm":
        model.design_logits(120, e_soft=1.0, num_models=1, ramp_recycles=True, **flags)
        model.design_soft(32, num_models=1, **flags)
        flags.update({"dropout": False, "save_best": True})
        model.design_soft(10, num_models=num_models, **flags)
        pssm = softmax(model.aux["seq"]["logits"], -1)

    else:
        # logits, soft, or hard optimization
        optimization_methods = {
            "logits": model.design_logits,
            "soft": model.design_soft,
            "hard": model.design_hard
        }

        if optimizer in optimization_methods:
            opt_fn = optimization_methods[optimizer]
            opt_fn(120, num_models=1, ramp_recycles=True, **flags)
            flags.update({"dropout": False, "save_best": True})
            opt_fn(10, num_models=num_models, **flags)
            pssm = softmax(model.aux["seq"]["logits"], -1)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    # Save results
    model.save_pdb(output_file)

    if verbose:
        print(f"Binder design complete! Saved to: {output_file}")
        if "log" in model._tmp.get("best", {}).get("aux", {}):
            print(f"Final metrics: {model._tmp['best']['aux']['log']}")

    # Get designed sequences
    sequences = model.get_seqs()

    results = {
        "sequences": sequences,
        "pdb_file": output_file,
        "target_len": model._target_len,
        "binder_len": model._binder_len,
        "pssm": pssm,
        "model": model
    }

    if "best" in model._tmp and "aux" in model._tmp["best"]:
        results["metrics"] = model._tmp["best"]["aux"].get("log", {})

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Design cyclic peptide binders for target proteins"
    )
    parser.add_argument("--pdb", type=str, help="Path to target protein PDB file")
    parser.add_argument("--pdb_code", type=str, help="4-letter PDB code to download")
    parser.add_argument("--target_chain", type=str, default="A",
                       help="Target protein chain ID (default: A)")
    parser.add_argument("--binder_len", type=int, default=14,
                       help="Length of binder peptide (default: 14)")
    parser.add_argument("--binder_seq", type=str,
                       help="Initial binder sequence (optional)")
    parser.add_argument("--hotspot", type=str,
                       help="Target hotspot residues (e.g., '1-10,12,15')")
    parser.add_argument("--target_flexible", action="store_true",
                       help="Allow target backbone flexibility")
    parser.add_argument("--use_multimer", action="store_true",
                       help="Use AlphaFold-multimer")
    parser.add_argument("--optimizer", type=str,
                       choices=["pssm_semigreedy", "3stage", "semigreedy", "pssm",
                               "logits", "soft", "hard"],
                       default="pssm_semigreedy",
                       help="Optimization method (default: pssm_semigreedy)")
    parser.add_argument("--num_recycles", type=int, default=0,
                       help="Number of AF2 recycles (default: 0)")
    parser.add_argument("--num_models", type=int, default=2,
                       help="Number of models to use (default: 2)")
    parser.add_argument("--output", type=str, default="cyclic_binder.pdb",
                       help="Output PDB file (default: cyclic_binder.pdb)")
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
            print("=== AfCycDesign: Cyclic Peptide Binder Design ===")
            print(f"Target: {pdb_file}")
            print(f"Target chain: {args.target_chain}")
            print(f"Binder length: {args.binder_len}")
            print(f"Optimizer: {args.optimizer}")
            print(f"Output: {args.output}")

        results = design_cyclic_peptide_binder(
            pdb_file=pdb_file,
            target_chain=args.target_chain,
            binder_len=args.binder_len,
            binder_seq=args.binder_seq,
            target_hotspot=args.hotspot,
            target_flexible=args.target_flexible,
            use_multimer=args.use_multimer,
            optimizer=args.optimizer,
            num_recycles=args.num_recycles,
            num_models=args.num_models,
            output_file=args.output,
            verbose=not args.quiet
        )

        if not args.quiet:
            print("\n=== Results ===")
            print(f"Target length: {results['target_len']}")
            print(f"Binder length: {results['binder_len']}")

            for i, seq in enumerate(results["sequences"]):
                print(f"Binder sequence {i+1}: {seq}")

            # Additional metrics if available
            if "metrics" in results:
                metrics = results["metrics"]
                if "plddt" in metrics:
                    print(f"pLDDT: {metrics['plddt']:.3f}")
                if "pae" in metrics:
                    print(f"PAE: {metrics['pae']:.3f}")
                if "i_con" in metrics:
                    print(f"Interface contacts: {metrics['i_con']:.3f}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())