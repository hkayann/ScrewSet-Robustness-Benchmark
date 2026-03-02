#!/usr/bin/env python3
"""Download ImageNet validation parquet files using huggingface_hub (with xet support)
then convert to ImageFolder format.
"""

import os
import sys
import json
import socket
from pathlib import Path

# Force IPv4
_orig = socket.getaddrinfo
def _ipv4(host, port, family=0, type=0, proto=0, flags=0):
    if family == 0: family = socket.AF_INET
    return _orig(host, port, family, type, proto, flags)
socket.getaddrinfo = _ipv4

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

DATA_DIR = Path("/home/hakan/ARCADE--Screwset/data")
CACHE_DIR = DATA_DIR / "imagenet-hf-cache"
OUTPUT_DIR = DATA_DIR / "imagenet-val"
CLASS_INDEX_PATH = DATA_DIR / "imagenet_class_index.json"


def load_class_mapping():
    if not CLASS_INDEX_PATH.exists():
        import urllib.request
        url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
        print(f"Downloading class index from {url}")
        urllib.request.urlretrieve(url, str(CLASS_INDEX_PATH))
    with open(CLASS_INDEX_PATH) as f:
        ci = json.load(f)
    return {int(k): v[0] for k, v in ci.items()}


def main():
    done_marker = OUTPUT_DIR / ".done"
    if done_marker.exists():
        count = sum(1 for _ in OUTPUT_DIR.rglob("*.JPEG"))
        print(f"Already done: {count} images in {OUTPUT_DIR}")
        return

    label_to_synset = load_class_mapping()
    print(f"Class mapping: {len(label_to_synset)} classes")

    # Step 1: Download all 14 parquet files
    print("\n=== Step 1: Download parquet files ===")
    parquet_paths = []
    for i in range(14):
        fname = f"data/validation-{i:05d}-of-00014.parquet"
        local_path = CACHE_DIR / fname

        # Check if already downloaded and valid
        if local_path.exists() and local_path.stat().st_size > 100_000_000:
            try:
                pf = pq.ParquetFile(str(local_path))
                print(f"[SKIP] {fname} ({local_path.stat().st_size/1e6:.0f} MB, {pf.metadata.num_rows} rows)")
                parquet_paths.append(local_path)
                continue
            except Exception:
                pass  # Re-download if invalid

        print(f"[DOWN] {fname} ...")
        path = hf_hub_download(
            repo_id='ILSVRC/imagenet-1k',
            filename=fname,
            repo_type='dataset',
            local_dir=str(CACHE_DIR)
        )
        parquet_paths.append(Path(path))
        print(f"  -> {os.path.getsize(path)/1e6:.0f} MB")

    # Step 2: Convert to ImageFolder
    print("\n=== Step 2: Convert to ImageFolder ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for synset in label_to_synset.values():
        (OUTPUT_DIR / synset).mkdir(exist_ok=True)

    total = 0
    for ppath in sorted(parquet_paths):
        print(f"Processing {ppath.name} ...", end=" ", flush=True)
        pf = pq.ParquetFile(str(ppath))

        for rg_idx in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg_idx)
            for i in range(len(table)):
                label = table.column("label")[i].as_py()
                img_data = table.column("image")[i].as_py()
                synset = label_to_synset[label]
                img_bytes = img_data["bytes"]

                total += 1
                fname = f"ILSVRC2012_val_{total:08d}.JPEG"
                out_path = OUTPUT_DIR / synset / fname

                if not out_path.exists():
                    with open(out_path, "wb") as f:
                        f.write(img_bytes)

        print(f"({total} cumulative)")

    with open(done_marker, "w") as f:
        f.write(f"Converted {total} images\n")

    print(f"\nDone! {total} images -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
