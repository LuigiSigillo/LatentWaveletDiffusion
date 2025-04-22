#!/bin/bash

# Check if the required arguments are provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <HF_TOKEN> <DOWNLOAD_DIR>"
  exit 1
fi

HF_TOKEN="$1"
DOWNLOAD_DIR="$2"

mkdir -p "$DOWNLOAD_DIR"  # Create the folder if it doesn't exist

for i in {0..127}; do 
  wget --header="Authorization: Bearer $HF_TOKEN" \
    "https://huggingface.co/datasets/laion/laion-high-resolution/resolve/main/part-$(printf "%05d" $i)-45914064-d424-4c1c-8d96-dc8125c645fb-c000.snappy.parquet?download=true" \
    -O "$DOWNLOAD_DIR/part-$(printf "%05d" $i).snappy.parquet"
done