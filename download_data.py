"""Download LEVIR-CD (256x256 pre-cropped) and lay it out the way
src/dataset.py expects:

    data/LEVIR-CD/
        train/A/*.png   train/B/*.png   train/label/*.png
        val/A/...       val/B/...       val/label/...
        test/A/...      test/B/...      test/label/...

Source: a public mirror of the standard LEVIR-CD split
(train=7120, val=1024, test=2048 pairs), which matches Table I
of the paper exactly.

Usage (from the project root, same folder as config.py):
    pip install datasets pillow
    python download_data.py
"""

from __future__ import annotations

from pathlib import Path

from datasets import load_dataset

HF_DATASET = "ericyu/LEVIRCD_Cropped256"
DEST_ROOT = Path("data/LEVIR-CD")

# HF split name -> local folder name used by dataset.py
SPLIT_MAP = {"train": "train", "val": "val", "test": "test"}


def save_split(hf_split_name: str, local_split_name: str, dataset) -> None:
    a_dir = DEST_ROOT / local_split_name / "A"
    b_dir = DEST_ROOT / local_split_name / "B"
    l_dir = DEST_ROOT / local_split_name / "label"
    for d in (a_dir, b_dir, l_dir):
        d.mkdir(parents=True, exist_ok=True)

    split = dataset[hf_split_name]
    n = len(split)
    print(f"Writing {n} pairs for split '{local_split_name}' -> {DEST_ROOT / local_split_name}")

    for i, example in enumerate(split):
        fname = f"{i:05d}.png"
        example["imageA"].save(a_dir / fname)
        example["imageB"].save(b_dir / fname)
        example["label"].save(l_dir / fname)

        if (i + 1) % 500 == 0 or (i + 1) == n:
            print(f"  {local_split_name}: {i + 1}/{n}")


def main() -> None:
    print(f"Downloading {HF_DATASET} from the Hugging Face Hub (first run only, cached after)...")
    dataset = load_dataset(HF_DATASET)
    print("Available splits:", list(dataset.keys()))

    for hf_split, local_split in SPLIT_MAP.items():
        if hf_split not in dataset:
            print(f"WARNING: split '{hf_split}' not found in dataset, skipping.")
            continue
        save_split(hf_split, local_split, dataset)

    print(f"Done. LEVIR-CD is ready at {DEST_ROOT.resolve()}")
    print("You can now run:  python -m src.evaluate --data data/LEVIR-CD")


if __name__ == "__main__":
    main()