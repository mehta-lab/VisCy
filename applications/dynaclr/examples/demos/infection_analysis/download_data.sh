#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: bash download_data.sh [OUTPUT_DIR]"
    echo ""
    echo "Download DynaCLR infection analysis demo data."
    echo ""
    echo "Arguments:"
    echo "  OUTPUT_DIR  Directory to download data into (default: ~/data/dynaclr/demo)"
    exit 0
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
fi

output_dir="${1:-$HOME/data/dynaclr/demo}"

mkdir -p "$output_dir"

echo "Downloading data to: $output_dir"

wget -m -np -nH --cut-dirs=6 -R "index.html*" \
    -P "$output_dir" \
    "https://public.czbiohub.org/comp.micro/viscy/DynaCLR_data/DENV/test/20240204_A549_DENV_ZIKV_timelapse/"

echo "Data downloaded successfully to: $output_dir"
