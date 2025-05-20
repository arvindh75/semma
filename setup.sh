#!/bin/bash

# Install prerequisites
echo "Installing requirements..."
# It is highly recommended to install PyTorch first, matching your CUDA version.
# The requirements.txt file specifies a version compatible with CUDA 11.8.
# Example: pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

pip install torch==2.1.0
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html # repace with correct torch and cuda versions

# Then install other requirements
pip install -r requirements.txt

echo ""
echo "IMPORTANT: The 'flash-attn' package can have complex installation dependencies related to your CUDA toolkit version."
echo "If you encounter issues with 'flash-attn', please refer to its official documentation for installation instructions specific to your environment."
echo ""

# Download and prepare fb_mid2name.tsv
echo "Downloading fb_mid2name.tsv..."
wget -O fb_mid2name.tsv 'https://drive.google.com/uc?id=0B52yRXcdpG6MaHA5ZW9CZ21MbVk'

echo "Unzipping fb_mid2name.tsv"

echo "Setup complete."
echo "Note: If fb_mid2name.tsv was downloaded as a zip archive, you might need to manually unzip it or adjust this script." 