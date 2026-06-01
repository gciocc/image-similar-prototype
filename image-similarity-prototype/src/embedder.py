"""Calcolo embedding immagini con OpenCLIP."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Union

import numpy as np
import open_clip
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import CLIP_MODEL, CLIP_PRETRAINED


class ImagePathDataset(Dataset):
    def __init__(self, paths: List[Path], preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        return self.preprocess(image), str(path)


class ClipEmbedder:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED
        )
        self.model.eval()
        self.model.to(self.device)
        self._dim: int | None = None

    @property
    def embedding_dim(self) -> int:
        if self._dim is None:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224, device=self.device)
                out = self.model.encode_image(dummy)
                self._dim = int(out.shape[-1])
        return self._dim

    @torch.inference_mode()
    def encode_paths(
        self,
        paths: Iterable[Union[str, Path]],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> tuple[np.ndarray, list[str]]:
        path_list = [Path(p) for p in paths]
        if not path_list:
            raise ValueError("Nessun percorso immagine fornito.")

        dataset = ImagePathDataset(path_list, self.preprocess)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.device == "cuda",
        )

        all_embeddings: list[np.ndarray] = []
        all_ids: list[str] = []

        iterator = loader
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(loader, desc="Embedding", unit="batch")

        for batch_images, batch_ids in iterator:
            batch_images = batch_images.to(self.device)
            features = self.model.encode_image(batch_images)
            features = features / features.norm(dim=-1, keepdim=True)
            all_embeddings.append(features.cpu().numpy().astype(np.float32))
            all_ids.extend(batch_ids)

        return np.vstack(all_embeddings), all_ids

    @torch.inference_mode()
    def encode_pil(self, image: Image.Image) -> np.ndarray:
        tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        features = self.model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)[0]
