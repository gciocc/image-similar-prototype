#!/usr/bin/env python3
"""
Fase 1 del prototipo: precalcola gli embedding per tutte le immagini del dataset.

Uso:
  python scripts/precompute_embeddings.py
  python scripts/precompute_embeddings.py --batch-size 128
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import IMAGES_DIR
from src.embedder import ClipEmbedder
from src.search import save_artifacts


def collect_images(images_dir: Path) -> list[Path]:
    extensions = {".png", ".jpg", ".jpeg", ".webp"}
    paths = sorted(
        p for p in images_dir.rglob("*") if p.suffix.lower() in extensions
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Precalcola embedding CLIP + indice FAISS")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=IMAGES_DIR,
        help=f"Cartella immagini (default: {IMAGES_DIR})",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Dispositivo per il modello CLIP",
    )
    args = parser.parse_args()

    if not args.images_dir.is_dir():
        print(
            f"Cartella immagini assente: {args.images_dir}\n"
            "Esegui prima: python scripts/download_dataset.py",
            file=sys.stderr,
        )
        sys.exit(1)

    paths = collect_images(args.images_dir)
    if not paths:
        print(f"Nessuna immagine in {args.images_dir}", file=sys.stderr)
        sys.exit(1)

    device = None if args.device == "auto" else args.device
    embedder = ClipEmbedder(device=device)
    print(f"Modello CLIP su {embedder.device}, dim={embedder.embedding_dim}")
    print(f"Elaborazione di {len(paths)} immagini…")

    embeddings, image_ids = embedder.encode_paths(
        paths, batch_size=args.batch_size, show_progress=True
    )
    save_artifacts(embeddings, image_ids)
    print(f"Salvati embedding ({embeddings.shape}) e indice FAISS.")


if __name__ == "__main__":
    main()
