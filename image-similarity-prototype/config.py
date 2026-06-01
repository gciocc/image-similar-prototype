"""Configurazione condivisa per download, embedding e ricerca."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Dataset Hugging Face: FFHQ thumbnails 128×128 (~70k immagini, ~2 GB)
HF_DATASET_ID = "nuwandaa/ffhq128"
HF_DATASET_SPLIT = "train"

# Percorsi locali
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
INDEX_DIR = DATA_DIR / "index"

EMBEDDINGS_NPY = EMBEDDINGS_DIR / "embeddings.npy"
IMAGE_IDS_JSON = EMBEDDINGS_DIR / "image_ids.json"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"

# Modello CLIP (bilanciato per prototipo su CPU)
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "openai"

# Ricerca
TOP_K_DEFAULT = 12
