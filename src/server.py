"""MCP Server for Cyclic Peptide Tools

Provides both synchronous and asynchronous (submit) APIs for all tools.
"""

from fastmcp import FastMCP
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import sys
import json
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
MCP_ROOT = SCRIPT_DIR.parent
SCRIPTS_DIR = MCP_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from jobs.manager import job_manager
from loguru import logger

# Create MCP server
mcp = FastMCP("cycpep-tools")

# ==============================================================================
# Job Management Tools (for async operations)
# ==============================================================================

@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """
    Get the status of a submitted cyclic peptide computation job.

    Args:
        job_id: The job ID returned from a submit_* function

    Returns:
        Dictionary with job status, timestamps, and any errors
    """
    return job_manager.get_job_status(job_id)

@mcp.tool()
def get_job_result(job_id: str) -> dict:
    """
    Get the results of a completed cyclic peptide computation job.

    Args:
        job_id: The job ID of a completed job

    Returns:
        Dictionary with the job results or error if not completed
    """
    return job_manager.get_job_result(job_id)

@mcp.tool()
def get_job_log(job_id: str, tail: int = 50) -> dict:
    """
    Get log output from a running or completed job.

    Args:
        job_id: The job ID to get logs for
        tail: Number of lines from end (default: 50, use 0 for all)

    Returns:
        Dictionary with log lines and total line count
    """
    return job_manager.get_job_log(job_id, tail)

@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """
    Cancel a running cyclic peptide computation job.

    Args:
        job_id: The job ID to cancel

    Returns:
        Success or error message
    """
    return job_manager.cancel_job(job_id)

@mcp.tool()
def list_jobs(status: Optional[str] = None) -> dict:
    """
    List all submitted cyclic peptide computation jobs.

    Args:
        status: Filter by status (pending, running, completed, failed, cancelled)

    Returns:
        List of jobs with their status
    """
    return job_manager.list_jobs(status)

# ==============================================================================
# Synchronous Tools (for fast operations < 10 min)
# ==============================================================================

@mcp.tool()
def predict_cyclic_structure(
    length: int,
    output_file: Optional[str] = None,
    rm_aa: Optional[str] = None,
    add_rg: bool = False,
    soft_iters: Optional[int] = None,
    quiet: bool = True
) -> dict:
    """
    Predict 3D structure of a cyclic peptide from scratch (hallucination).

    Fast operation - returns results immediately for typical peptides (5-15 residues).

    Args:
        length: Length of the peptide to generate (5-50 residues)
        output_file: Optional output PDB file path
        rm_aa: Amino acids to exclude (default: "C", format: "C,M,P")
        add_rg: Add radius of gyration constraint for compact structures
        soft_iters: Number of soft iterations (default: 50, use 20 for faster testing)
        quiet: Suppress verbose output (default: True)

    Returns:
        Dictionary with generated sequence, structure file path, and quality metrics
    """
    try:
        # Import here to avoid loading dependencies unless needed
        sys.path.insert(0, str(SCRIPTS_DIR))
        from predict_cyclic_structure import run_predict_cyclic_structure

        # Set up arguments
        kwargs = {
            'length': length,
            'output_file': output_file,
            'quiet': quiet
        }

        if rm_aa is not None:
            kwargs['rm_aa'] = rm_aa
        if add_rg:
            kwargs['add_rg'] = True
        if soft_iters is not None:
            kwargs['soft_iters'] = soft_iters

        result = run_predict_cyclic_structure(**kwargs)
        return {"status": "success", **result}

    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"Structure prediction failed: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
def design_cyclic_sequence(
    input_file: str,
    output_file: Optional[str] = None,
    chain: str = "A",
    positions: Optional[str] = None,
    iterations: Optional[int] = None,
    quiet: bool = True
) -> dict:
    """
    Redesign amino acid sequence for a given cyclic peptide backbone structure.

    Fast operation - typically completes within 5-10 minutes for small peptides.

    Args:
        input_file: Path to input PDB file with backbone structure
        output_file: Optional output PDB file path for designed structure
        chain: Chain ID to design (default: "A")
        positions: Specific positions to design (format: "1-5,10", default: all)
        iterations: Number of design iterations (default: 100)
        quiet: Suppress verbose output (default: True)

    Returns:
        Dictionary with designed sequence, structure file path, and design metrics
    """
    try:
        # Import here to avoid loading dependencies unless needed
        sys.path.insert(0, str(SCRIPTS_DIR))
        from design_cyclic_sequence import run_design_cyclic_sequence

        # Set up arguments
        kwargs = {
            'input_file': input_file,
            'output_file': output_file,
            'chain': chain,
            'quiet': quiet
        }

        if positions is not None:
            kwargs['positions'] = positions
        if iterations is not None:
            kwargs['iterations'] = iterations

        result = run_design_cyclic_sequence(**kwargs)
        return {"status": "success", **result}

    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"Sequence design failed: {e}")
        return {"status": "error", "error": str(e)}

# ==============================================================================
# Submit Tools (for long-running operations > 10 min)
# ==============================================================================

@mcp.tool()
def submit_cyclic_binder_design(
    target_file: str,
    binder_len: int,
    output_file: Optional[str] = None,
    target_chain: str = "A",
    hotspot: Optional[str] = None,
    iterations: Optional[int] = None,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit a cyclic peptide binder design job for background processing.

    This task may take 15-30+ minutes depending on target protein size.
    Use get_job_status() to monitor progress and get_job_result() to retrieve results.

    Args:
        target_file: Path to target protein PDB file
        binder_len: Length of binder peptide to design (6-20 residues)
        output_file: Optional output PDB file path for designed binder
        target_chain: Target protein chain ID (default: "A")
        hotspot: Specific binding residues (format: "1-5,10", default: automatic)
        iterations: Number of design iterations (default: 100)
        job_name: Optional name for job tracking

    Returns:
        Dictionary with job_id for tracking. Use:
        - get_job_status(job_id) to check progress
        - get_job_result(job_id) to get results when completed
        - get_job_log(job_id) to see execution logs
    """
    script_path = str(SCRIPTS_DIR / "design_cyclic_binder.py")

    args = {
        "target": target_file,
        "binder_len": binder_len,
        "target_chain": target_chain,
        "quiet": True
    }

    if output_file:
        args["output"] = output_file
    if hotspot:
        args["hotspot"] = hotspot
    if iterations:
        args["iterations"] = iterations

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or f"binder_design_{binder_len}mer"
    )

@mcp.tool()
def submit_large_structure_prediction(
    length: int,
    output_file: Optional[str] = None,
    rm_aa: Optional[str] = None,
    add_rg: bool = False,
    soft_iters: int = 100,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit a large cyclic peptide structure prediction job for background processing.

    For large peptides (>15 residues) or high-accuracy runs (>100 iterations).
    Use this for production-quality structure generation.

    Args:
        length: Length of the peptide to generate (typically 15-50 residues)
        output_file: Optional output PDB file path
        rm_aa: Amino acids to exclude (format: "C,M,P")
        add_rg: Add radius of gyration constraint for compact structures
        soft_iters: Number of soft iterations (default: 100 for high quality)
        job_name: Optional name for job tracking

    Returns:
        Dictionary with job_id for tracking
    """
    script_path = str(SCRIPTS_DIR / "predict_cyclic_structure.py")

    args = {
        "length": length,
        "soft_iters": soft_iters,
        "quiet": True
    }

    if output_file:
        args["output"] = output_file
    if rm_aa:
        args["rm_aa"] = rm_aa
    if add_rg:
        args["add_rg"] = True

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or f"structure_prediction_{length}mer"
    )

@mcp.tool()
def submit_large_sequence_design(
    input_file: str,
    output_file: Optional[str] = None,
    chain: str = "A",
    positions: Optional[str] = None,
    iterations: int = 200,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit a large cyclic peptide sequence design job for background processing.

    For complex backbones or high-accuracy design runs (>200 iterations).

    Args:
        input_file: Path to input PDB file with backbone structure
        output_file: Optional output PDB file path
        chain: Chain ID to design (default: "A")
        positions: Specific positions to design (format: "1-5,10", default: all)
        iterations: Number of design iterations (default: 200 for high accuracy)
        job_name: Optional name for job tracking

    Returns:
        Dictionary with job_id for tracking
    """
    script_path = str(SCRIPTS_DIR / "design_cyclic_sequence.py")

    args = {
        "input": input_file,
        "chain": chain,
        "iterations": iterations,
        "quiet": True
    }

    if output_file:
        args["output"] = output_file
    if positions:
        args["positions"] = positions

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or f"sequence_design_{chain}"
    )

# ==============================================================================
# Batch Processing Tools
# ==============================================================================

@mcp.tool()
def submit_batch_structure_prediction(
    lengths: List[int],
    output_dir: Optional[str] = None,
    rm_aa: Optional[str] = None,
    add_rg: bool = False,
    soft_iters: int = 50,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit batch structure prediction for multiple cyclic peptide lengths.

    Generates multiple cyclic peptides in a single job. Useful for:
    - Virtual library generation
    - Length optimization studies
    - Parallel peptide design

    Args:
        lengths: List of peptide lengths to generate (e.g., [8, 10, 12])
        output_dir: Directory for output files (default: auto-generated)
        rm_aa: Amino acids to exclude for all peptides
        add_rg: Add radius of gyration constraint
        soft_iters: Number of iterations per structure (default: 50)
        job_name: Optional name for the batch job

    Returns:
        Dictionary with job_id for tracking the batch job
    """
    # Create a batch script dynamically
    script_path = str(SCRIPTS_DIR / "predict_cyclic_structure.py")

    # We'll run multiple jobs in sequence - this could be optimized with a proper batch script
    batch_args = {
        "lengths": ",".join(map(str, lengths)),
        "soft_iters": soft_iters,
        "quiet": True
    }

    if output_dir:
        batch_args["output_dir"] = output_dir
    if rm_aa:
        batch_args["rm_aa"] = rm_aa
    if add_rg:
        batch_args["add_rg"] = True

    return job_manager.submit_job(
        script_path=script_path,
        args=batch_args,
        job_name=job_name or f"batch_prediction_{len(lengths)}_peptides"
    )

# ==============================================================================
# Utility Tools
# ==============================================================================

@mcp.tool()
def validate_cyclic_peptide_file(file_path: str) -> dict:
    """
    Validate a cyclic peptide PDB file structure.

    Args:
        file_path: Path to PDB file to validate

    Returns:
        Dictionary with validation results and structural information
    """
    try:
        pdb_path = Path(file_path)
        if not pdb_path.exists():
            return {"status": "error", "error": f"File not found: {file_path}"}

        if not pdb_path.is_file():
            return {"status": "error", "error": f"Not a file: {file_path}"}

        if pdb_path.stat().st_size == 0:
            return {"status": "error", "error": f"Empty file: {file_path}"}

        # Basic PDB validation
        chains = set()
        atom_count = 0
        residue_count = 0

        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    atom_count += 1
                    chain = line[21:22]
                    chains.add(chain)
                elif line.startswith('HETATM'):
                    atom_count += 1

        return {
            "status": "success",
            "file_path": str(pdb_path),
            "file_size_bytes": pdb_path.stat().st_size,
            "chains": list(chains),
            "atom_count": atom_count,
            "estimated_residues": atom_count // 10,  # Rough estimate
            "valid_pdb": atom_count > 0
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.tool()
def get_server_info() -> dict:
    """
    Get information about the MCP server and available tools.

    Returns:
        Dictionary with server information, tool counts, and status
    """
    return {
        "status": "success",
        "server_name": "cycpep-tools",
        "version": "1.0.0",
        "scripts_directory": str(SCRIPTS_DIR),
        "jobs_directory": str(job_manager.jobs_dir),
        "available_tools": {
            "sync_tools": [
                "predict_cyclic_structure",
                "design_cyclic_sequence",
                "validate_cyclic_peptide_file"
            ],
            "submit_tools": [
                "submit_cyclic_binder_design",
                "submit_large_structure_prediction",
                "submit_large_sequence_design",
                "submit_batch_structure_prediction"
            ],
            "job_management": [
                "get_job_status",
                "get_job_result",
                "get_job_log",
                "cancel_job",
                "list_jobs"
            ]
        },
        "api_types": {
            "sync": "For fast operations completing in <10 minutes",
            "submit": "For long-running operations >10 minutes with background processing"
        }
    }

# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    mcp.run()