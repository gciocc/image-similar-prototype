#!/usr/bin/env python3
"""
Scarica le immagini FFHQ 128×128 da Hugging Face (nuwandaa/ffhq128).

Il repo espone sia shard Parquet (consigliato, anche per subset) sia uno ZIP
(thumbnails128x128.zip, ~2 GB).

Uso:
  python scripts/download_dataset.py
  python scripts/download_dataset.py --max-images 500
  python scripts/download_dataset.py --from-zip
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import HF_DATASET_ID, HF_DATASET_SPLIT, IMAGES_DIR
from tqdm import tqdm

ZIP_FILENAME = "thumbnails128x128.zip"


def _save_pil_image(img, out_path: Path) -> None:
    if hasattr(img, "save"):
        img.save(out_path)
    elif isinstance(img, bytes):
        out_path.write_bytes(img)
    else:
        shutil.copy2(img, out_path)


def _image_column(row: dict) -> object:
    for key in ("image", "img", "bytes"):
        if key in row:
            return row[key]
    for val in row.values():
        if hasattr(val, "save"):
            return val
    raise KeyError(f"Nessuna colonna immagine in: {list(row.keys())}")


def download_parquet(max_images: int | None, output_dir: Path) -> int:
    from datasets import load_dataset

    print(f"Caricamento Parquet da {HF_DATASET_ID}…")
    ds = load_dataset(HF_DATASET_ID, split=HF_DATASET_SPLIT)
    n = len(ds) if max_images is None else min(max_images, len(ds))
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(n), desc="Salvataggio", unit="img"):
        row = ds[i]
        img = _image_column(row)
        out_path = output_dir / f"{i:05d}.png"
        _save_pil_image(img, out_path)
    return n


def download_zip(max_images: int | None, output_dir: Path) -> int:
    from huggingface_hub import hf_hub_download

    print(f"Download {ZIP_FILENAME} da Hugging Face (~2 GB)…")
    zip_path = Path(
        hf_hub_download(
            repo_id=HF_DATASET_ID,
            filename=ZIP_FILENAME,
            repo_type="dataset",
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    extract_root = output_dir.parent / "_zip_extract"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True)

    print("Estrazione archivio…")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)

    png_files = sorted(extract_root.rglob("*.png"))
    if not png_files:
        raise RuntimeError(f"Nessun PNG trovato in {extract_root}")

    if max_images is not None:
        png_files = png_files[:max_images]

    for i, src in enumerate(tqdm(png_files, desc="Copia immagini", unit="img")):
        dst = output_dir / f"{i:05d}.png"
        shutil.copy2(src, dst)

    shutil.rmtree(extract_root, ignore_errors=True)
    return len(png_files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scarica FFHQ 128×128 da Hugging Face")
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limita il numero di immagini (default: tutte ~70k)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=IMAGES_DIR,
        help=f"Cartella di destinazione (default: {IMAGES_DIR})",
    )
    parser.add_argument(
        "--from-zip",
        action="store_true",
        help="Scarica ed estrae thumbnails128x128.zip invece dei Parquet",
    )
    args = parser.parse_args()

    try:
        if args.from_zip:
            count = download_zip(args.max_images, args.output_dir)
        else:
            count = download_parquet(args.max_images, args.output_dir)
    except ImportError:
        print("Installa le dipendenze: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        if args.from_zip:
            raise
        print(f"Parquet non disponibile ({exc}). Riprovo con --from-zip…")
        count = download_zip(args.max_images, args.output_dir)

    print(f"Fatto: {count} immagini in {args.output_dir}")


if __name__ == "__main__":
    main()
