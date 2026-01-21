#!/bin/bash
# Example runner script for cyclic peptide structure prediction
# Usage: ./run_example.sh [quick|production|compact] [gpu_id]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
CONFIG_TYPE="${1:-quick}"
GPU_ID="${2:-0}"

# Set environment
export ALPHAFOLD_DATA_DIR="$BASE_DIR/params"

# Select config file
case "$CONFIG_TYPE" in
    quick)
        CONFIG="$SCRIPT_DIR/data/config_quick_test.json"
        LENGTH=8
        ;;
    production)
        CONFIG="$SCRIPT_DIR/data/config_production.json"
        LENGTH=8
        ;;
    compact)
        CONFIG="$SCRIPT_DIR/data/config_compact_peptide.json"
        LENGTH=10
        ;;
    *)
        echo "Usage: $0 [quick|production|compact] [gpu_id]"
        echo "  quick      - Fast test with reduced iterations (default)"
        echo "  production - High-quality with more iterations"
        echo "  compact    - Compact structure with Rg constraint"
        echo "  gpu_id     - GPU device ID (default: 0, use -1 for CPU)"
        exit 1
        ;;
esac

# Create output directory
mkdir -p "$SCRIPT_DIR/outputs"

# Generate output filename with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="$SCRIPT_DIR/outputs/${CONFIG_TYPE}_${LENGTH}mer_${TIMESTAMP}.pdb"

echo "=== Cyclic Peptide Structure Prediction ==="
echo "Config: $CONFIG_TYPE"
echo "Length: $LENGTH residues"
echo "Output: $OUTPUT"

# Build command
CMD="$BASE_DIR/env/bin/python $BASE_DIR/scripts/predict_cyclic_structure.py"
CMD="$CMD --length $LENGTH"
CMD="$CMD --output $OUTPUT"
CMD="$CMD --config $CONFIG"

if [ "$GPU_ID" -ge 0 ] 2>/dev/null; then
    CMD="$CMD --gpu $GPU_ID"
    echo "GPU: $GPU_ID"
else
    CMD="$CMD --cpu"
    echo "Device: CPU"
fi

echo ""
echo "Running: $CMD"
echo ""

# Run prediction
$CMD

echo ""
echo "Done! Output saved to: $OUTPUT"
