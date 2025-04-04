export HF_TOKEN=""
export DOWNLOAD_DIR="/mnt/share/Luigi/Documents/URAE/dataset/laion_high_resolution_parquet"

mkdir -p "$DOWNLOAD_DIR"  # Create the folder if it doesn't exist

for i in {0..127}; do 
  wget --header="Authorization: Bearer $HF_TOKEN" \
    "https://huggingface.co/datasets/laion/laion-high-resolution/resolve/main/part-$(printf "%05d" $i)-45914064-d424-4c1c-8d96-dc8125c645fb-c000.snappy.parquet?download=true" \
    -O "$DOWNLOAD_DIR/part-$(printf "%05d" $i).snappy.parquet"
done