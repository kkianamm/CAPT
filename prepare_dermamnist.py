"""
prepare_dermamnist.py
=====================
Download and arrange DermaMNIST so it matches BiomedCoOp's preprocessing exactly.

BiomedCoOp's "preprocessing" for DermaMNIST is:
  1. the fixed train/val/test split shipped as ``data/DermaMNIST/split_DermaMNIST.json``
     (7007 / 1003 / 2005 images, 7 classes), consumed by ``datasets/dermamnist.py``;
  2. deterministic few-shot subsampling (``generate_fewshot_dataset``) keyed by seed;
  3. base/novel subsampling (first ceil(7/2)=4 classes = base, last 3 = novel);
  4. image transforms: bicubic resize to 224, CLIP mean/std normalization,
     ``random_resized_crop`` for training (see the config yaml).

Steps 2-4 happen automatically inside the dataset loader / config. This script
only fetches the images and puts the split json in place. Run it from the repo
root (the BiomedCoOp clone that now also contains the CAPT files).

    python prepare_dermamnist.py --root data

Result:
    data/DermaMNIST/
    |-- DermaMNIST/            <- class sub-folders with images
    |-- split_DermaMNIST.json  <- BiomedCoOp's fixed split
"""

import argparse
import os
import shutil
import zipfile

URL = "https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/DermaMNIST.zip"


def download(url, dst):
    if os.path.exists(dst):
        print(f"{dst} already present, skipping download.")
        return
    try:
        from huggingface_hub import hf_hub_download
        print("Downloading via huggingface_hub ...")
        path = hf_hub_download(repo_id="TahaKoleilat/BiomedCoOp",
                               filename="DermaMNIST.zip", repo_type="dataset")
        shutil.copy(path, dst)
        return
    except Exception as e:
        print(f"huggingface_hub failed ({e}); falling back to streaming download.")
    import requests
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"Saved {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data", help="dataset root (matches --root of train_capt.py)")
    args = ap.parse_args()

    os.makedirs(args.root, exist_ok=True)
    zip_path = os.path.join(args.root, "DermaMNIST.zip")
    download(URL, zip_path)

    print("Unzipping ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(args.root)

    ds_dir = os.path.join(args.root, "DermaMNIST")
    img_dir = os.path.join(ds_dir, "DermaMNIST")
    split_dst = os.path.join(ds_dir, "split_DermaMNIST.json")

    # The BiomedCoOp repo ships the canonical split; prefer it if the zip lacks one.
    repo_split = os.path.join("data", "DermaMNIST", "split_DermaMNIST.json")
    if not os.path.exists(split_dst) and os.path.exists(repo_split):
        shutil.copy(repo_split, split_dst)
        print(f"Copied canonical split to {split_dst}")

    ok = os.path.isdir(img_dir) and os.path.exists(split_dst)
    print("\nLayout check:")
    print(f"  images dir : {img_dir}  ->  {'OK' if os.path.isdir(img_dir) else 'MISSING'}")
    print(f"  split json : {split_dst}  ->  {'OK' if os.path.exists(split_dst) else 'MISSING'}")
    print("\nDone." if ok else "\nPlease verify the extracted layout above.")


if __name__ == "__main__":
    main()
