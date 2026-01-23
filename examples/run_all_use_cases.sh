#!/bin/bash
# ==============================================================================
# AfCycDesign: Run All Use Case Examples
# ==============================================================================
# This script provides example commands for all four primary use cases.
#
# Usage:
#   ./run_all_use_cases.sh [use_case] [gpu_id]
#
# Examples:
#   ./run_all_use_cases.sh 0 0        # Run Use Case 0 on GPU 0
#   ./run_all_use_cases.sh 1 1        # Run Use Case 1 on GPU 1
#   ./run_all_use_cases.sh all 0      # Run all use cases on GPU 0
#   ./run_all_use_cases.sh 2 -1       # Run Use Case 2 on CPU
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
USE_CASE="${1:-help}"
GPU_ID="${2:-0}"

# Set environment
export ALPHAFOLD_DATA_DIR="$BASE_DIR/params"

# Create output directory
OUTPUT_DIR="$SCRIPT_DIR/outputs"
mkdir -p "$OUTPUT_DIR"

# Timestamp for unique filenames
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# GPU configuration
if [ "$GPU_ID" -ge 0 ] 2>/dev/null; then
    GPU_FLAG="--gpu $GPU_ID"
    DEVICE_INFO="GPU $GPU_ID"
else
    GPU_FLAG="--cpu"
    DEVICE_INFO="CPU"
fi

# ==============================================================================
# Use Case 0: Structure Prediction
# ==============================================================================
run_use_case_0() {
    echo "============================================================"
    echo "Use Case 0: Structure Prediction"
    echo "============================================================"
    echo "Predicting 3D structures of known cyclic peptide sequences"
    echo "Device: $DEVICE_INFO"
    echo ""

    # Example 0a: Benchmark sequence 1JBL (SFTI-1)
    echo "--- Example 0a: Benchmark 1JBL (GRCTKSIPPICFPD) ---"
    $BASE_DIR/env/bin/python "$SCRIPT_DIR/use_case_0_structure_prediction.py" \
        --benchmark 1JBL \
        --num_models 3 \
        --num_recycles 3 \
        --output "$OUTPUT_DIR/use_case_0_1JBL_${TIMESTAMP}.pdb" \
        --save_json

    echo ""

    # Example 0b: Benchmark sequence 2MW0
    echo "--- Example 0b: Benchmark 2MW0 (FRLLNYYA) ---"
    $BASE_DIR/env/bin/python "$SCRIPT_DIR/use_case_0_structure_prediction.py" \
        --benchmark 2MW0 \
        --num_models 3 \
        --num_recycles 3 \
        --output "$OUTPUT_DIR/use_case_0_2MW0_${TIMESTAMP}.pdb" \
        --save_json

    echo ""

    # Example 0c: Custom sequence
    echo "--- Example 0c: Custom sequence (RVKDGYPF) ---"
    $BASE_DIR/env/bin/python "$SCRIPT_DIR/use_case_0_structure_prediction.py" \
        --sequence "RVKDGYPF" \
        --num_models 1 \
        --num_recycles 3 \
        --output "$OUTPUT_DIR/use_case_0_custom_${TIMESTAMP}.pdb"

    echo ""
    echo "Use Case 0 complete! Outputs saved to: $OUTPUT_DIR"
}

# ==============================================================================
# Use Case 1: Fixed Backbone Design (Fixbb)
# ==============================================================================
run_use_case_1() {
    echo "============================================================"
    echo "Use Case 1: Fixed Backbone Sequence Design"
    echo "============================================================"
    echo "Redesigning sequences for cyclic peptide backbone structures"
    echo "Device: $DEVICE_INFO"
    echo ""

    # Example 1a: Redesign 1JBL backbone
    echo "--- Example 1a: Redesign 1JBL backbone (14 residues) ---"
    $BASE_DIR/env/bin/python "$SCRIPT_DIR/use_case_1_cyclic_fixbb_design.py" \
        --pdb "$SCRIPT_DIR/data/structures/1JBL_chainA.pdb" \
        --chain A \
        --stage_iters 30 30 10 \
        --output "$OUTPUT_DIR/use_case_1_1JBL_redesign_${TIMESTAMP}.pdb"

    echo ""

    # Example 1b: Redesign with compactness constraint
    echo "--- Example 1b: Redesign with Rg constraint ---"
    $BASE_DIR/env/bin/python "$SCRIPT_DIR/use_case_1_cyclic_fixbb_design.py" \
        --pdb "$SCRIPT_DIR/data/structures/1JBL_chainA.pdb" \
        --chain A \
        --add_rg \
        --rg_weight 0.1 \
        --stage_iters 30 30 10 \
        --output "$OUTPUT_DIR/use_case_1_1JBL_compact_${TIMESTAMP}.pdb"

    echo ""

    # Example 1c: Quick test with simple backbone
    echo "--- Example 1c: Quick test with test_backbone.pdb ---"
    $BASE_DIR/env/bin/python "$SCRIPT_DIR/use_case_1_cyclic_fixbb_design.py" \
        --pdb "$SCRIPT_DIR/data/test_backbone.pdb" \
        --chain A \
        --stage_iters 20 20 5 \
        --output "$OUTPUT_DIR/use_case_1_test_${TIMESTAMP}.pdb"

    echo ""
    echo "Use Case 1 complete! Outputs saved to: $OUTPUT_DIR"
}

# ==============================================================================
# Use Case 2: De Novo Hallucination
# ==============================================================================
run_use_case_2() {
    echo "============================================================"
    echo "Use Case 2: De Novo Hallucination"
    echo "============================================================"
    echo "Generating novel cyclic peptide structures and sequences"
    echo "Device: $DEVICE_INFO"
    echo ""

    # Example 2a: 8-mer hallucination (quick)
    echo "--- Example 2a: Quick 8-mer hallucination ---"
    $BASE_DIR/env/bin/python "$SCRIPT_DIR/use_case_2_cyclic_hallucination.py" \
        --length 8 \
        --rm_aa "C" \
        --soft_iters 30 \
        --stage_iters 30 30 10 \
        --plddt_threshold 0.7 \
        --output "$OUTPUT_DIR/use_case_2_8mer_${TIMESTAMP}.pdb"

    echo ""

    # Example 2b: 12-mer hallucination (standard)
    echo "--- Example 2b: Standard 12-mer hallucination ---"
    $BASE_DIR/env/bin/python "$SCRIPT_DIR/use_case_2_cyclic_hallucination.py" \
        --length 12 \
        --rm_aa "C,M" \
        --soft_iters 50 \
        --stage_iters 50 50 10 \
        --plddt_threshold 0.9 \
        --output "$OUTPUT_DIR/use_case_2_12mer_${TIMESTAMP}.pdb"

    echo ""

    # Example 2c: Compact 10-mer with Rg constraint
    echo "--- Example 2c: Compact 10-mer with Rg constraint ---"
    $BASE_DIR/env/bin/python "$SCRIPT_DIR/use_case_2_cyclic_hallucination.py" \
        --length 10 \
        --rm_aa "C" \
        --add_rg \
        --rg_weight 0.15 \
        --soft_iters 50 \
        --stage_iters 50 50 10 \
        --output "$OUTPUT_DIR/use_case_2_10mer_compact_${TIMESTAMP}.pdb"

    echo ""
    echo "Use Case 2 complete! Outputs saved to: $OUTPUT_DIR"
}

# ==============================================================================
# Use Case 3: Binder Design (Motif Grafting)
# ==============================================================================
run_use_case_3() {
    echo "============================================================"
    echo "Use Case 3: Cyclic Peptide Binder Design"
    echo "============================================================"
    echo "Designing cyclic peptide binders for protein targets"
    echo "Device: $DEVICE_INFO"
    echo ""

    # Example 3a: Design binder for MDM2
    echo "--- Example 3a: Design 12-mer binder for MDM2 ---"
    $BASE_DIR/env/bin/python "$SCRIPT_DIR/use_case_3_cyclic_binder_design.py" \
        --pdb "$SCRIPT_DIR/data/structures/4HFZ_MDM2.pdb" \
        --target_chain A \
        --binder_len 12 \
        --optimizer "pssm_semigreedy" \
        --num_models 1 \
        --ipae_threshold 0.15 \
        --output "$OUTPUT_DIR/use_case_3_MDM2_binder_${TIMESTAMP}.pdb"

    echo ""

    # Example 3b: Design binder for Keap1
    echo "--- Example 3b: Design 14-mer binder for Keap1 ---"
    $BASE_DIR/env/bin/python "$SCRIPT_DIR/use_case_3_cyclic_binder_design.py" \
        --pdb "$SCRIPT_DIR/data/structures/2FLU_Keap1.pdb" \
        --target_chain X \
        --binder_len 14 \
        --optimizer "pssm_semigreedy" \
        --num_models 1 \
        --ipae_threshold 0.15 \
        --output "$OUTPUT_DIR/use_case_3_Keap1_binder_${TIMESTAMP}.pdb"

    echo ""

    # Example 3c: Design binder with initial motif (p53 binding motif)
    echo "--- Example 3c: Design binder with p53 motif seed ---"
    $BASE_DIR/env/bin/python "$SCRIPT_DIR/use_case_3_cyclic_binder_design.py" \
        --pdb "$SCRIPT_DIR/data/structures/4HFZ_MDM2.pdb" \
        --target_chain A \
        --binder_seq "FSDLWKLLPEN" \
        --optimizer "3stage" \
        --num_models 1 \
        --ipae_threshold 0.11 \
        --output "$OUTPUT_DIR/use_case_3_MDM2_motif_${TIMESTAMP}.pdb"

    echo ""
    echo "Use Case 3 complete! Outputs saved to: $OUTPUT_DIR"
}

# ==============================================================================
# Help
# ==============================================================================
show_help() {
    echo "============================================================"
    echo "AfCycDesign: Use Case Examples"
    echo "============================================================"
    echo ""
    echo "Usage: $0 [use_case] [gpu_id]"
    echo ""
    echo "Use Cases:"
    echo "  0    - Structure Prediction (predict 3D structure from sequence)"
    echo "  1    - Fixed Backbone Design (redesign sequence for backbone)"
    echo "  2    - De Novo Hallucination (generate novel structure + sequence)"
    echo "  3    - Binder Design (design cyclic peptide binders)"
    echo "  all  - Run all use cases"
    echo "  help - Show this help message"
    echo ""
    echo "GPU ID:"
    echo "  0, 1, 2, ... - Use specific GPU"
    echo "  -1           - Use CPU"
    echo ""
    echo "Examples:"
    echo "  $0 0 0       # Run Structure Prediction on GPU 0"
    echo "  $0 1 1       # Run Fixbb Design on GPU 1"
    echo "  $0 2 -1      # Run Hallucination on CPU"
    echo "  $0 all 0     # Run all use cases on GPU 0"
    echo ""
    echo "============================================================"
    echo "Individual Commands:"
    echo "============================================================"
    echo ""
    echo "# Use Case 0: Structure Prediction"
    echo "python use_case_0_structure_prediction.py --benchmark 1JBL --num_models 5 --num_recycles 6"
    echo "python use_case_0_structure_prediction.py --sequence \"RVKDGYPF\" --output predicted.pdb"
    echo ""
    echo "# Use Case 1: Fixed Backbone Design"
    echo "python use_case_1_cyclic_fixbb_design.py --pdb data/structures/1JBL_chainA.pdb --chain A"
    echo "python use_case_1_cyclic_fixbb_design.py --pdb data/test_backbone.pdb --add_rg"
    echo ""
    echo "# Use Case 2: De Novo Hallucination"
    echo "python use_case_2_cyclic_hallucination.py --length 12 --rm_aa \"C\" --plddt_threshold 0.9"
    echo "python use_case_2_cyclic_hallucination.py --length 10 --add_rg --rg_weight 0.15"
    echo ""
    echo "# Use Case 3: Binder Design"
    echo "python use_case_3_cyclic_binder_design.py --pdb data/structures/4HFZ_MDM2.pdb --target_chain A --binder_len 12"
    echo "python use_case_3_cyclic_binder_design.py --pdb data/structures/2FLU_Keap1.pdb --target_chain X --binder_len 14"
    echo ""
}

# ==============================================================================
# Main
# ==============================================================================
echo ""
echo "============================================================"
echo "AfCycDesign Use Case Examples"
echo "============================================================"
echo "Base directory: $BASE_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Device: $DEVICE_INFO"
echo "============================================================"
echo ""

case "$USE_CASE" in
    0)
        run_use_case_0
        ;;
    1)
        run_use_case_1
        ;;
    2)
        run_use_case_2
        ;;
    3)
        run_use_case_3
        ;;
    all)
        run_use_case_0
        echo ""
        run_use_case_1
        echo ""
        run_use_case_2
        echo ""
        run_use_case_3
        ;;
    help|*)
        show_help
        ;;
esac

echo ""
echo "============================================================"
echo "Done!"
echo "============================================================"
