#!/bin/bash
# Quick Setup Script for ColabDesign MCP (afcycdesign)
# ColabDesign: Making Protein Design accessible to all via Google Colab
# Includes TrDesign, AfDesign, ProteinMPNN, and RFdiffusion tools
# Source: https://github.com/sokrypton/ColabDesign

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Setting up ColabDesign MCP ==="

# Step 1: Create Python environment
echo "[1/6] Creating Python 3.10 environment..."
(command -v mamba >/dev/null 2>&1 && mamba create -p ./env python=3.10 pip -y) || \
(command -v conda >/dev/null 2>&1 && conda create -p ./env python=3.10 pip -y) || \
(echo "Warning: Neither mamba nor conda found, creating venv instead" && python3 -m venv ./env)

# Step 2: Install JAX with CUDA support
echo "[2/6] Installing JAX with CUDA support..."
./env/bin/pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Step 3: Install ColabDesign
echo "[3/6] Installing ColabDesign v1.1.1..."
./env/bin/pip install git+https://github.com/sokrypton/ColabDesign.git@v1.1.1

# Step 4: Install RDKit
echo "[4/6] Installing RDKit..."
(command -v mamba >/dev/null 2>&1 && mamba install -p ./env -c conda-forge rdkit -y) || \
(command -v conda >/dev/null 2>&1 && conda install -p ./env -c conda-forge rdkit -y) || \
./env/bin/pip install rdkit

# Step 5: Install utility packages
echo "[5/6] Installing utility packages..."
./env/bin/pip install --force-reinstall --no-cache-dir loguru click tqdm

# Step 6: Install fastmcp
echo "[6/6] Installing fastmcp..."
./env/bin/pip install --ignore-installed fastmcp

echo ""
echo "=== ColabDesign MCP Setup Complete ==="
echo "To run the MCP server: ./env/bin/python src/server.py"
