#!/usr/bin/env python3
"""
Script: predict_cyclic_structure.py
Description: Predict 3D structure of cyclic peptides from scratch (hallucination)

Original Use Case: examples/use_case_2_cyclic_hallucination.py
Dependencies Removed: None (all essential for core functionality)

Usage:
    # Using command-line arguments
    python scripts/predict_cyclic_structure.py --length 8 --output structure.pdb
    python scripts/predict_cyclic_structure.py --length 10 --rm_aa "C,M" --add_rg --output compact.pdb

    # Using YAML config file (recommended)
    python scripts/predict_cyclic_structure.py --config example/data/predict_8mer.yaml
    python scripts/predict_cyclic_structure.py --config example/data/predict_12mer_production.yaml --gpu 0

GPU Usage:
    python scripts/predict_cyclic_structure.py --length 8 --output structure.pdb --gpu 0
    python scripts/predict_cyclic_structure.py --config config.yaml --gpu 1 --gpu_mem_fraction 0.8

Example:
    python scripts/predict_cyclic_structure.py --length 8 --output examples/data/predicted_8mer.pdb
    python scripts/predict_cyclic_structure.py --config example/data/predict_compact_peptide.yaml
"""

# ==============================================================================
# GPU Configuration (MUST be set before importing JAX)
# ==============================================================================
import os
import argparse
import sys
from pathlib import Path


def get_nvidia_lib_paths() -> list:
    """
    Get nvidia pip package library paths for LD_LIBRARY_PATH.
    This is needed for JAX to find CUDA libraries like cuSPARSE.

    Returns:
        List of library paths.
    """
    lib_paths = []

    # Find nvidia packages in site-packages
    possible_nvidia_dirs = []

    # Method 1: Check relative to script location (for conda/mamba envs)
    script_dir = Path(__file__).parent.resolve()
    env_site_packages = script_dir.parent / "env" / "lib" / "python3.10" / "site-packages" / "nvidia"
    if env_site_packages.exists():
        possible_nvidia_dirs.append(env_site_packages)

    # Method 2: Find nvidia package location via Python import
    try:
        import nvidia.cusparse as _cusparse
        if hasattr(_cusparse, '__file__') and _cusparse.__file__:
            nvidia_pkg_dir = Path(_cusparse.__file__).parent.parent
            if nvidia_pkg_dir.exists() and nvidia_pkg_dir not in possible_nvidia_dirs:
                possible_nvidia_dirs.append(nvidia_pkg_dir)
        elif hasattr(_cusparse, '__path__'):
            # Namespace package - get path from __path__
            for p in _cusparse.__path__:
                nvidia_pkg_dir = Path(p).parent
                if nvidia_pkg_dir.exists() and nvidia_pkg_dir not in possible_nvidia_dirs:
                    possible_nvidia_dirs.append(nvidia_pkg_dir)
    except (ImportError, AttributeError):
        pass

    # Method 3: Check in sys.path for site-packages
    for p in sys.path:
        nvidia_dir = Path(p) / "nvidia"
        if nvidia_dir.exists() and nvidia_dir not in possible_nvidia_dirs:
            possible_nvidia_dirs.append(nvidia_dir)

    # Libraries subdirectories to add
    nvidia_lib_subdirs = [
        "cuda_runtime/lib", "nvjitlink/lib", "cublas/lib",
        "cufft/lib", "cusparse/lib", "cusolver/lib", "cudnn/lib", "nccl/lib"
    ]

    # Collect all lib paths
    for nvidia_dir in possible_nvidia_dirs:
        for lib_subdir in nvidia_lib_subdirs:
            lib_path = nvidia_dir / lib_subdir
            if lib_path.exists() and str(lib_path) not in lib_paths:
                lib_paths.append(str(lib_path))

    return lib_paths


def check_gpu_env_and_reexec():
    """
    Check if LD_LIBRARY_PATH is set for GPU. If not, re-exec with correct env.
    This ensures CUDA libraries are available before JAX loads.
    """
    # Skip if explicitly using CPU
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return

    # Check if we've already re-exec'd
    if os.environ.get("_CYCPEP_GPU_ENV_SET") == "1":
        return

    # Get required nvidia lib paths
    nvidia_paths = get_nvidia_lib_paths()
    if not nvidia_paths:
        return  # No nvidia libs found, skip

    # Check if LD_LIBRARY_PATH already has these paths
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    needs_update = False
    for p in nvidia_paths:
        if p not in current_ld:
            needs_update = True
            break

    if needs_update:
        # Re-exec with updated LD_LIBRARY_PATH
        new_ld = ":".join(nvidia_paths)
        if current_ld:
            new_ld = f"{new_ld}:{current_ld}"
        os.environ["LD_LIBRARY_PATH"] = new_ld
        os.environ["_CYCPEP_GPU_ENV_SET"] = "1"

        # Re-exec this script
        os.execv(sys.executable, [sys.executable] + sys.argv)


def configure_gpu(gpu_id: int = None, mem_fraction: float = None,
                  preallocate: bool = True) -> dict:
    """
    Configure GPU settings for JAX. Must be called BEFORE importing JAX.

    Args:
        gpu_id: GPU device ID to use (0, 1, etc.). None for CPU or auto-select.
        mem_fraction: Fraction of GPU memory to use (0.0-1.0). None for default.
        preallocate: Whether to preallocate GPU memory (default: True).
                    Set to False for more flexible memory usage.

    Returns:
        Dict with configuration status and device info.
    """
    config_info = {
        "gpu_requested": gpu_id,
        "mem_fraction": mem_fraction,
        "preallocate": preallocate,
        "env_vars_set": []
    }

    if gpu_id is not None:
        # Set specific GPU device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        config_info["env_vars_set"].append(f"CUDA_VISIBLE_DEVICES={gpu_id}")

    if mem_fraction is not None:
        # Limit GPU memory fraction
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)
        config_info["env_vars_set"].append(f"XLA_PYTHON_CLIENT_MEM_FRACTION={mem_fraction}")

    if not preallocate:
        # Disable memory preallocation for more flexible usage
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        config_info["env_vars_set"].append("XLA_PYTHON_CLIENT_PREALLOCATE=false")

    return config_info


def get_device_info() -> dict:
    """
    Get information about available compute devices.
    Must be called AFTER importing JAX.

    Returns:
        Dict with device information.
    """
    import jax

    devices = jax.devices()
    device_info = {
        "num_devices": len(devices),
        "devices": [],
        "default_backend": jax.default_backend(),
        "using_gpu": jax.default_backend() == "gpu"
    }

    for d in devices:
        device_info["devices"].append({
            "id": d.id,
            "platform": d.platform,
            "device_kind": d.device_kind if hasattr(d, 'device_kind') else str(d),
        })

    return device_info


def parse_gpu_args():
    """
    Pre-parse GPU-related arguments before full argument parsing.
    This allows GPU configuration before JAX import.
    """
    # Simple pre-parse for GPU args only
    gpu_id = None
    mem_fraction = None
    preallocate = True

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] in ('--gpu', '-g') and i + 1 < len(args):
            try:
                gpu_id = int(args[i + 1])
            except ValueError:
                pass
            i += 2
        elif args[i] == '--gpu_mem_fraction' and i + 1 < len(args):
            try:
                mem_fraction = float(args[i + 1])
            except ValueError:
                pass
            i += 2
        elif args[i] == '--no_gpu_preallocate':
            preallocate = False
            i += 1
        elif args[i] == '--cpu':
            # Force CPU mode
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            gpu_id = -1  # Signal CPU mode
            i += 1
        else:
            i += 1

    return gpu_id, mem_fraction, preallocate


# Pre-parse GPU args and configure environment before JAX import
_gpu_id, _mem_fraction, _preallocate = parse_gpu_args()
_gpu_config = configure_gpu(_gpu_id, _mem_fraction, _preallocate)

# Re-exec with proper LD_LIBRARY_PATH if needed for GPU support
check_gpu_env_and_reexec()

# Store nvidia lib paths info
_gpu_config["nvidia_lib_paths"] = get_nvidia_lib_paths()

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Essential scientific packages for cyclic peptide prediction
import numpy as np
import jax.numpy as jnp
import jax
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.af.alphafold.common import residue_constants

# Get device info after JAX import
_device_info = get_device_info()

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
# YAML/JSON Config Loading
# ==============================================================================
def load_config_file(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.

    YAML config format:
    ```yaml
    name: "job_name"
    description: "Job description"

    peptide:
      length: 8
      exclude_amino_acids: "C,M"

    constraints:
      add_rg: true
      rg_weight: 0.15
      offset_type: 2

    optimization:
      num_recycles: 0
      soft_iters: 50
      stage_iters: [50, 50, 10]

    loss_weights:
      pae: 1.0
      plddt: 1.0
      con: 0.5

    output:
      file: "output.pdb"
      save_metadata: true

    gpu:
      device: 0
      mem_fraction: 0.9
    ```

    Args:
        config_path: Path to YAML or JSON config file

    Returns:
        Flattened configuration dictionary compatible with run_predict_cyclic_structure
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load raw config
    with open(config_file) as f:
        if config_file.suffix.lower() in ('.yaml', '.yml'):
            try:
                import yaml
                raw_config = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML is required for YAML config files. Install with: pip install pyyaml")
        else:
            raw_config = json.load(f)

    if raw_config is None:
        raw_config = {}

    # Flatten nested YAML structure to flat config
    config = {}

    # Job metadata (for tracking)
    if 'name' in raw_config:
        config['job_name'] = raw_config['name']
    if 'description' in raw_config:
        config['job_description'] = raw_config['description']

    # Peptide section
    peptide = raw_config.get('peptide', {})
    if 'length' in peptide:
        config['length'] = peptide['length']
    if 'sequence' in peptide and peptide['sequence']:
        config['sequence'] = peptide['sequence']
    if 'exclude_amino_acids' in peptide:
        config['rm_aa'] = peptide['exclude_amino_acids']

    # Constraints section
    constraints = raw_config.get('constraints', {})
    if 'add_rg' in constraints:
        config['add_rg'] = constraints['add_rg']
    if 'rg_weight' in constraints:
        config['rg_weight'] = constraints['rg_weight']
    if 'offset_type' in constraints:
        config['offset_type'] = constraints['offset_type']

    # Optimization section
    optimization = raw_config.get('optimization', {})
    if 'num_recycles' in optimization:
        config['num_recycles'] = optimization['num_recycles']
    if 'soft_iters' in optimization:
        config['soft_iters'] = optimization['soft_iters']
    if 'stage_iters' in optimization:
        config['stage_iters'] = optimization['stage_iters']

    # Loss weights section
    if 'loss_weights' in raw_config:
        config['loss_weights'] = raw_config['loss_weights']

    # Output section
    output = raw_config.get('output', {})
    if 'file' in output and output['file']:
        config['output_file'] = output['file']
    if 'save_metadata' in output:
        config['save_metadata'] = output['save_metadata']

    # GPU section
    gpu = raw_config.get('gpu', {})
    if 'device' in gpu and gpu['device'] is not None:
        config['gpu_device'] = gpu['device']
    if 'mem_fraction' in gpu and gpu['mem_fraction'] is not None:
        config['gpu_mem_fraction'] = gpu['mem_fraction']
    if 'preallocate' in gpu:
        config['gpu_preallocate'] = gpu['preallocate']

    # Also support flat JSON format for backwards compatibility
    for key in ['rm_aa', 'offset_type', 'add_rg', 'rg_weight', 'num_recycles',
                'soft_iters', 'stage_iters', 'contact_cutoff', 'loss_weights']:
        if key in raw_config and key not in config:
            config[key] = raw_config[key]

    # Store raw config for metadata
    config['_raw_config'] = raw_config

    return config


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
def validate_sequence(sequence: str) -> str:
    """
    Validate and normalize a peptide sequence.

    Args:
        sequence: Amino acid sequence (1-letter codes)

    Returns:
        Normalized uppercase sequence

    Raises:
        ValueError: If sequence contains invalid amino acids
    """
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    sequence = sequence.upper().strip()

    invalid = set(sequence) - valid_aa
    if invalid:
        raise ValueError(f"Invalid amino acids in sequence: {invalid}. "
                        f"Valid amino acids: {''.join(sorted(valid_aa))}")

    if len(sequence) < 5:
        raise ValueError("Sequence must be at least 5 residues")
    if len(sequence) > 50:
        print(f"Warning: Very long sequences (>50) may be challenging to predict", file=sys.stderr)

    return sequence


def run_predict_cyclic_structure(
    length: int = None,
    sequence: str = None,
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Predict cyclic peptide 3D structure.

    Two modes:
    1. Hallucination mode (sequence=None): Generate both sequence and structure
    2. Sequence mode (sequence provided): Predict structure for given sequence

    Args:
        length: Length of peptide to generate (required for hallucination mode)
        sequence: Amino acid sequence for structure prediction (optional)
                  If provided, predicts structure for this sequence with cyclic constraints
        output_file: Path to save PDB file (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - result: Predicted sequences and structure data
            - output_file: Path to output PDB file (if saved)
            - metadata: Execution metadata
            - metrics: Quality metrics (pLDDT, PAE, contacts)

    Example (hallucination):
        >>> result = run_predict_cyclic_structure(length=8, output_file="output.pdb")
        >>> print(f"Sequence: {result['result']['sequences'][0]}")

    Example (from sequence):
        >>> result = run_predict_cyclic_structure(sequence="RVKDGYPF", output_file="output.pdb")
        >>> print(f"pLDDT: {result['metrics']['plddt']:.3f}")
    """
    # Setup configuration
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Determine mode and validate inputs
    if sequence:
        # Sequence mode: predict structure for given sequence
        sequence = validate_sequence(sequence)
        length = len(sequence)
        mode = "sequence"
    else:
        # Hallucination mode: generate sequence and structure
        if length is None:
            raise ValueError("Either 'length' or 'sequence' must be provided")
        validate_inputs(length, config.get("rm_aa", "C"))
        mode = "hallucination"

    # Clear any previous models
    clear_mem()

    # Initialize AlphaFold model for hallucination protocol
    # (works for both modes - we just fix the sequence in sequence mode)
    af_model = mk_afdesign_model(
        protocol="hallucination",
        num_recycles=config.get("num_recycles", 0)
    )

    # Prepare inputs
    if mode == "sequence":
        # In sequence mode, don't exclude any amino acids (pass None, not empty string)
        af_model.prep_inputs(length=length)
    else:
        # In hallucination mode, exclude specified amino acids
        rm_aa = config.get("rm_aa", "C")
        af_model.prep_inputs(length=length, rm_aa=rm_aa)

    # Add cyclic offset for head-to-tail cyclization
    add_cyclic_offset(af_model, offset_type=config.get("offset_type", 2))

    # Optionally add radius of gyration loss for compact structures
    if config.get("add_rg", False):
        add_rg_loss(af_model, weight=config.get("rg_weight", 0.1))

    # Initialize model
    af_model.restart()

    if mode == "sequence":
        # Sequence mode: predict structure for fixed sequence
        # Set the sequence and lock it (don't allow design to change it)
        af_model.set_seq(seq=sequence)

        # Configure contact loss
        af_model.set_opt("con", binary=True,
                        cutoff=config.get("contact_cutoff", 21.6875),
                        num=af_model._len, seqsep=0)

        # Set loss weights
        weights = config.get("loss_weights", DEFAULT_CONFIG["loss_weights"])
        af_model.set_weights(**weights)

        # For structure prediction, we run AlphaFold with the fixed sequence
        # Use predict() to get structure without changing sequence
        num_recycles = config.get("num_recycles", 1)
        af_model.predict(verbose=True)

    else:
        # Hallucination mode: generate sequence and structure
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

    # Build metadata
    run_metadata = {
        "length": af_model._len,
        "config": config,
        "mode": mode,
        "protocol": "hallucination",  # AlphaFold protocol used
        "device": {
            "backend": _device_info["default_backend"],
            "using_gpu": _device_info["using_gpu"],
            "devices": _device_info["devices"]
        }
    }

    # Add input sequence info for sequence mode
    if mode == "sequence":
        run_metadata["input_sequence"] = sequence

    # Save output if requested
    output_path = None
    if output_file:
        output_path = Path(output_file)
        af_model.save_pdb(str(output_path))
        save_output(str(output_path), sequences, metrics, run_metadata)

    return {
        "result": {
            "sequences": sequences,
            "model": af_model
        },
        "output_file": str(output_path) if output_path else None,
        "metrics": metrics,
        "metadata": run_metadata
    }


# ==============================================================================
# CLI Interface
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--length', '-l', type=int,
                       help='Length of peptide to generate (5-50 residues). Required unless sequence or config provided.')
    parser.add_argument('--sequence', '-s', type=str,
                       help='Amino acid sequence for structure prediction (e.g., "RVKDGYPF"). '
                            'If provided, predicts structure for this cyclic sequence.')
    parser.add_argument('--output', '-o', type=str,
                       help='Output PDB file path')
    parser.add_argument('--config', '-c', type=str,
                       help='Config file (YAML or JSON). See example/data/*.yaml for examples.')

    # Core parameters
    parser.add_argument('--rm_aa', type=str, default="C",
                       help='Amino acids to exclude in hallucination mode (comma-separated, default: C)')
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

    # GPU parameters (pre-parsed, but included for help text)
    gpu_group = parser.add_argument_group('GPU options')
    gpu_group.add_argument('--gpu', '-g', type=int, metavar='ID',
                          help='GPU device ID to use (0, 1, etc.). Omit for auto-select.')
    gpu_group.add_argument('--gpu_mem_fraction', type=float, metavar='FRAC',
                          help='Fraction of GPU memory to use (0.0-1.0, e.g., 0.8 for 80%%)')
    gpu_group.add_argument('--no_gpu_preallocate', action="store_true",
                          help='Disable GPU memory preallocation (more flexible but slower)')
    gpu_group.add_argument('--cpu', action="store_true",
                          help='Force CPU mode (ignore available GPUs)')

    parser.add_argument('--quiet', action="store_true",
                       help='Suppress verbose output')

    args = parser.parse_args()

    # Load config if provided (supports both YAML and JSON)
    file_config = {}
    if args.config:
        file_config = load_config_file(args.config)

    # Determine sequence (CLI > config file)
    sequence = args.sequence or file_config.get('sequence')

    # Determine peptide length (CLI > config file > from sequence)
    if sequence:
        length = len(sequence)
        mode = "sequence"
    else:
        length = args.length or file_config.get('length')
        mode = "hallucination"
        if length is None:
            parser.error("Either --length, --sequence, or config with peptide.length/peptide.sequence is required")

    # Determine output path (CLI > config file > auto-generate)
    output_path = args.output or file_config.get('output_file')
    if not output_path:
        # Auto-generate output path based on job name, sequence, or length
        if 'job_name' in file_config:
            job_name = file_config['job_name']
        elif sequence:
            job_name = f"cyclic_{sequence[:8]}{'...' if len(sequence) > 8 else ''}"
        else:
            job_name = f"cyclic_{length}mer"
        output_path = f"outputs/{job_name}.pdb"

    # Prepare config by merging: DEFAULT < file_config < CLI overrides
    config = dict(DEFAULT_CONFIG)

    # Apply file config values
    for key in ['rm_aa', 'offset_type', 'add_rg', 'rg_weight', 'num_recycles',
                'soft_iters', 'hard_iters', 'stage_iters', 'contact_cutoff', 'loss_weights']:
        if key in file_config:
            config[key] = file_config[key]

    # Apply CLI overrides (only if explicitly set, not default values)
    if args.rm_aa != "C":
        config["rm_aa"] = args.rm_aa
    if args.offset_type != 2:
        config["offset_type"] = args.offset_type
    if args.add_rg:
        config["add_rg"] = True
    if args.rg_weight != 0.1:
        config["rg_weight"] = args.rg_weight
    if args.num_recycles != 0:
        config["num_recycles"] = args.num_recycles
    if args.soft_iters != 50:
        config["soft_iters"] = args.soft_iters
    if args.stage_iters != [50, 50, 10]:
        config["stage_iters"] = args.stage_iters

    # Store job metadata
    if 'job_name' in file_config:
        config['job_name'] = file_config['job_name']
    if 'job_description' in file_config:
        config['job_description'] = file_config['job_description']
    if '_raw_config' in file_config:
        config['_raw_config'] = file_config['_raw_config']

    try:
        if not args.quiet:
            print("=== Cyclic Peptide Structure Prediction ===")
            if 'job_name' in config:
                print(f"Job: {config['job_name']}")
            if 'job_description' in config:
                print(f"Description: {config['job_description']}")
            print(f"Mode: {mode}")
            if sequence:
                print(f"Sequence: {sequence}")
            print(f"Length: {length}")
            if mode == "hallucination":
                print(f"Excluded AAs: {config.get('rm_aa', 'C')}")
            print(f"Output: {output_path}")
            if config.get('add_rg'):
                print(f"Radius of gyration constraint: weight={config.get('rg_weight', 0.1)}")
            if args.config:
                print(f"Config: {args.config}")

            # Display device info
            print(f"\n--- Device Configuration ---")
            print(f"Backend: {_device_info['default_backend'].upper()}")
            if _device_info['using_gpu']:
                for dev in _device_info['devices']:
                    print(f"Device {dev['id']}: {dev['device_kind']} ({dev['platform']})")
            else:
                print("Running on CPU")
            if _gpu_config['env_vars_set']:
                print(f"GPU config: {', '.join(_gpu_config['env_vars_set'])}")
            print()

        # Ensure output directory exists
        output_dir = Path(output_path).parent
        if output_dir and not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        # Run prediction
        result = run_predict_cyclic_structure(
            length=length if not sequence else None,
            sequence=sequence,
            output_file=output_path,
            config=config
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
        import traceback
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())