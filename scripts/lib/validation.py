"""
Input validation utilities for cyclic peptide MCP scripts.

These functions are extracted and simplified from repository code to provide
essential validation without external dependencies.
"""

from pathlib import Path
from typing import Union, Optional, List, Set
import re


def validate_peptide_length(length: int, min_length: int = 5, max_length: int = 50) -> None:
    """
    Validate peptide length is within acceptable range.

    Args:
        length: Peptide length to validate
        min_length: Minimum allowed length (default: 5)
        max_length: Maximum allowed length (default: 50)

    Raises:
        ValueError: If length is outside acceptable range
    """
    if length < min_length:
        raise ValueError(f"Peptide length must be at least {min_length} residues, got {length}")
    if length > max_length:
        raise ValueError(f"Peptide length must be at most {max_length} residues, got {length}")


def validate_amino_acids(aa_string: str, exclude_invalid: bool = False) -> str:
    """
    Validate amino acid string contains only valid single-letter codes.

    Args:
        aa_string: String of amino acids (e.g., "C,M" or "CM")
        exclude_invalid: If True, remove invalid characters; if False, raise error

    Returns:
        Cleaned amino acid string

    Raises:
        ValueError: If invalid amino acids found and exclude_invalid=False
    """
    # Standard amino acid single-letter codes
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")

    # Clean input string
    clean_aa = aa_string.replace(",", "").replace(" ", "").replace("-", "").upper()

    # Find invalid amino acids
    invalid_aa = set(clean_aa) - valid_aa

    if invalid_aa:
        if exclude_invalid:
            # Remove invalid characters
            clean_aa = "".join(aa for aa in clean_aa if aa in valid_aa)
        else:
            raise ValueError(f"Invalid amino acids: {invalid_aa}. "
                           f"Valid amino acids: {sorted(valid_aa)}")

    return clean_aa


def validate_pdb_file(pdb_file: Union[str, Path]) -> Path:
    """
    Validate PDB file exists, is readable, and has content.

    Args:
        pdb_file: Path to PDB file

    Returns:
        Validated Path object

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is invalid
    """
    pdb_path = Path(pdb_file)

    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    if not pdb_path.is_file():
        raise ValueError(f"Not a regular file: {pdb_path}")

    if pdb_path.stat().st_size == 0:
        raise ValueError(f"Empty PDB file: {pdb_path}")

    # Check if file has PDB content (basic check)
    try:
        with open(pdb_path, 'r') as f:
            first_lines = [f.readline() for _ in range(10)]
            has_pdb_content = any(
                line.startswith(('ATOM', 'HETATM', 'MODEL', 'HEADER'))
                for line in first_lines
            )
            if not has_pdb_content:
                raise ValueError(f"File does not appear to be a valid PDB file: {pdb_path}")
    except Exception as e:
        raise ValueError(f"Error reading PDB file {pdb_path}: {e}")

    return pdb_path


def validate_chain_id(pdb_file: Path, chain_id: str) -> None:
    """
    Validate that specified chain ID exists in the PDB file.

    Args:
        pdb_file: Validated PDB file path
        chain_id: Chain ID to validate

    Raises:
        ValueError: If chain not found in PDB
    """
    chains_found = set()

    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    if len(line) > 21:  # Ensure line is long enough
                        chains_found.add(line[21])
    except Exception as e:
        raise ValueError(f"Error reading PDB file for chain validation: {e}")

    if chain_id not in chains_found:
        available = sorted(chains_found) if chains_found else ["none"]
        raise ValueError(f"Chain '{chain_id}' not found in PDB. "
                        f"Available chains: {available}")


def parse_position_string(position_str: Optional[str]) -> Optional[List[int]]:
    """
    Parse position specification string to list of residue numbers.

    Args:
        position_str: String like "1-5,10,15-20" or None

    Returns:
        List of position indices (0-based) or None for all positions

    Raises:
        ValueError: If position string format is invalid

    Examples:
        >>> parse_position_string("1-3,5")
        [0, 1, 2, 4]
        >>> parse_position_string("10")
        [9]
        >>> parse_position_string(None)
        None
    """
    if not position_str:
        return None

    positions = []

    try:
        for part in position_str.split(','):
            part = part.strip()
            if not part:
                continue

            if '-' in part:
                # Range specification
                range_parts = part.split('-')
                if len(range_parts) != 2:
                    raise ValueError(f"Invalid range format: '{part}'. Use 'start-end'")

                start, end = map(int, range_parts)
                if start > end:
                    raise ValueError(f"Invalid range: start ({start}) > end ({end})")
                if start < 1:
                    raise ValueError(f"Position numbers must be >= 1, got {start}")

                positions.extend(range(start-1, end))  # Convert to 0-based

            else:
                # Single position
                pos = int(part)
                if pos < 1:
                    raise ValueError(f"Position numbers must be >= 1, got {pos}")
                positions.append(pos - 1)  # Convert to 0-based

    except ValueError as e:
        if "invalid literal" in str(e):
            raise ValueError(f"Invalid position string format: '{position_str}'. "
                           f"Use format like '1-5,10,15-20'")
        else:
            raise

    return sorted(set(positions))


def validate_config_dict(config: dict, required_keys: Optional[List[str]] = None,
                        allowed_keys: Optional[List[str]] = None) -> None:
    """
    Validate configuration dictionary structure.

    Args:
        config: Configuration dictionary to validate
        required_keys: Keys that must be present
        allowed_keys: Keys that are allowed (if None, all keys allowed)

    Raises:
        ValueError: If validation fails
    """
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a dictionary, got {type(config)}")

    # Check required keys
    if required_keys:
        missing_keys = set(required_keys) - set(config.keys())
        if missing_keys:
            raise ValueError(f"Missing required config keys: {sorted(missing_keys)}")

    # Check allowed keys
    if allowed_keys:
        invalid_keys = set(config.keys()) - set(allowed_keys)
        if invalid_keys:
            raise ValueError(f"Invalid config keys: {sorted(invalid_keys)}. "
                           f"Allowed keys: {sorted(allowed_keys)}")


def validate_output_path(output_path: Union[str, Path], create_dirs: bool = True) -> Path:
    """
    Validate and prepare output path.

    Args:
        output_path: Desired output file path
        create_dirs: Whether to create parent directories

    Returns:
        Validated Path object

    Raises:
        ValueError: If path is invalid
        OSError: If cannot create directories
    """
    output_path = Path(output_path)

    # Check parent directory
    parent = output_path.parent
    if not parent.exists():
        if create_dirs:
            try:
                parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise OSError(f"Cannot create output directory {parent}: {e}")
        else:
            raise ValueError(f"Output directory does not exist: {parent}")

    # Check if parent is writable
    if not parent.is_dir():
        raise ValueError(f"Output parent path is not a directory: {parent}")

    return output_path