"""Indice FAISS e ricerca per similarità coseno."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from config import EMBEDDINGS_NPY, FAISS_INDEX_PATH, IMAGE_IDS_JSON, INDEX_DIR


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Indice inner-product su vettori L2-normalizzati (= similarità coseno)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


def save_artifacts(
    embeddings: np.ndarray,
    image_ids: list[str],
    embeddings_path: Path = EMBEDDINGS_NPY,
    ids_path: Path = IMAGE_IDS_JSON,
    index_path: Path = FAISS_INDEX_PATH,
) -> None:
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(embeddings_path, embeddings.astype(np.float32))
    ids_path.write_text(json.dumps(image_ids, ensure_ascii=False), encoding="utf-8")

    index = build_faiss_index(embeddings)
    faiss.write_index(index, str(index_path))


class SimilaritySearch:
    def __init__(
        self,
        embeddings_path: Path = EMBEDDINGS_NPY,
        ids_path: Path = IMAGE_IDS_JSON,
        index_path: Path = FAISS_INDEX_PATH,
    ):
        if not embeddings_path.exists() or not ids_path.exists() or not index_path.exists():
            raise FileNotFoundError(
                "Indice non trovato. Esegui prima:\n"
                "  python scripts/precompute_embeddings.py"
            )
        self.embeddings = np.load(embeddings_path)
        self.image_ids: List[str] = json.loads(ids_path.read_text(encoding="utf-8"))
        self.index = faiss.read_index(str(index_path))

    def search(self, query: np.ndarray, top_k: int = 12) -> List[Tuple[str, float]]:
        q = query.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        scores, indices = self.index.search(q, top_k)
        results: List[Tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            results.append((self.image_ids[idx], float(score)))
        return results
