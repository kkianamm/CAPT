"""
prepare_retina.py
=================
Download and arrange the RETINA dataset so it matches BiomedCoOp's handling
exactly (same as datasets/retina.py expects).

BiomedCoOp's RETINA "preprocessing/protocol":
  1. the fixed split shipped as data/RETINA/split_RETINA.json
     (2108 / 841 / 1268 images, 4 classes: diabetic retinopathy, glaucoma,
      cataract, normal retina);
  2. deterministic seed-keyed few-shot subsampling (generate_fewshot_dataset);
  3. base/novel split: base = first ceil(4/2)=2 classes
     (diabetic retinopathy, glaucoma), novel = last 2 (cataract, normal retina);
  4. image transforms: bicubic resize to 224, CLIP mean/std normalization,
     random_resized_crop on train -- set in the config yaml.

Steps 2-4 happen automatically in the loader / config. This script only fetches
the images and puts the split json in place. Run from the repo root.

    python prepare_retina.py --root data

Result:
    data/RETINA/
    |-- RETINA/               <- class sub-folders with images
    |-- split_RETINA.json     <- BiomedCoOp's fixed split (already shipped)
"""

import argparse
import os
import shutil
import zipfile

URL = "https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main/RETINA.zip"


def download(url, dst):
    if os.path.exists(dst):
        print(f"{dst} already present, skipping download.")
        return
    try:
        from huggingface_hub import hf_hub_download
        print("Downloading via huggingface_hub ...")
        path = hf_hub_download(repo_id="TahaKoleilat/BiomedCoOp",
                               filename="RETINA.zip", repo_type="dataset")
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
    zip_path = os.path.join(args.root, "RETINA.zip")
    download(URL, zip_path)

    print("Unzipping ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(args.root)

    ds_dir = os.path.join(args.root, "RETINA")
    img_dir = os.path.join(ds_dir, "RETINA")
    split_dst = os.path.join(ds_dir, "split_RETINA.json")

    # The repo already ships the canonical split; keep it if the zip lacks one.
    repo_split = os.path.join("data", "RETINA", "split_RETINA.json")
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
