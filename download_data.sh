#!/bin/bash
# Download graph datasets from HGCN (Chami et al., NeurIPS 2019)

set -e

if [ -d "data" ]; then
    echo "data/ already exists, skipping download."
    exit 0
fi

echo "Cloning HGCN to get datasets..."
git clone --depth 1 https://github.com/HazyResearch/hgcn.git /tmp/hgcn_data

echo "Copying datasets..."
cp -r /tmp/hgcn_data/data ./data

echo "Cleaning up..."
rm -rf /tmp/hgcn_data

echo "Done. Datasets available:"
ls data/
# Expected: airport  cora  disease_lp  disease_nc  pubmed
