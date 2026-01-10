#!/bin/bash
# Setup script for ISIS3 in a separate conda environment
# This is needed because ISIS3 requires Python 3.10, but the main project uses Python 3.12

set -e  # Exit on error

echo "=========================================="
echo "ISIS3 Setup for CTX Processing"
echo "=========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create isis3 environment with Python 3.10
echo ""
echo "Creating isis3 conda environment with Python 3.10..."
conda create -y -n isis3 -c conda-forge -c usgs-astrogeology python=3.10 isis

echo ""
echo "=========================================="
echo "ISIS3 Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the ISIS3 environment:"
echo "   conda activate isis3"
echo ""
echo "2. Download SPICE data for MRO/CTX (~500 MB):"
echo "   downloadIsisData base --mro"
echo ""
echo "3. Set environment variables (add to ~/.bashrc for persistence):"
echo "   export ISISROOT=\$CONDA_PREFIX"
echo "   export ISISDATA=\$HOME/isis3data"
echo ""
echo "4. Test the installation:"
echo "   which mroctx2isis"
echo "   conda activate base  # Return to base environment"
echo "   pixi run python test_ctx_download.py"
echo ""
echo "Note: The CTX downloader will automatically find ISIS3 commands"
echo "      even when running from the pixi environment!"
echo ""
