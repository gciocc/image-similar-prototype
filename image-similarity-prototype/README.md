# Prototipo: ricerca per similarità visiva

Sistema in tre fasi, come da specifica:

| Fase | Cosa fa | Script |
|------|---------|--------|
| **0** | Scarica il dataset (~70k volti 128×128) | `scripts/download_dataset.py` |
| **1** | Precalcola embedding CLIP + indice FAISS | `scripts/precompute_embeddings.py` |
| **2–3** | Embedding della foto utente + UI risultati | `app/main.py` |

## Dataset

Usiamo **[FFHQ thumbnails 128×128](https://huggingface.co/datasets/nuwandaa/ffhq128)** su Hugging Face (`nuwandaa/ffhq128`):

- ~**70.000** immagini PNG **128×128**
- ~**2 GB** totali (adatto a un prototipo locale)
- Stesso contenuto del repo NVIDIA [ffhq-dataset](https://github.com/NVlabs/ffhq-dataset) (`--thumbs`), ma scaricabile via `datasets` senza Google Drive

> **Licenza**: uso non commerciale (CC BY-NC-SA). Non usare per face recognition in produzione (vedi README NVIDIA/HF).

## Requisiti

- Python 3.10+
- ~3 GB spazio disco (immagini + embedding + modello CLIP)
- Consigliato: GPU per la fase 1 (funziona anche su CPU, più lenta)

## Installazione

```bash
cd image-similarity-prototype
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso rapido (prova con 500 immagini)

```bash
# 1) Scarica un subset (pochi minuti)
python scripts/download_dataset.py --max-images 500

# 2) Precalcola embedding (~1–5 min su CPU con 500 img)
python scripts/precompute_embeddings.py

# 3) Avvia l'interfaccia
python app/main.py
```

Apri http://127.0.0.1:7860, carica una foto e vedi le immagini più simili del catalogo.

## Dataset completo (~70k)

```bash
python scripts/download_dataset.py          # ~2 GB, può richiedere tempo
python scripts/precompute_embeddings.py     # decine di minuti su CPU / meno su GPU
python app/main.py
```

## Struttura cartelle

```
image-similarity-prototype/
├── app/main.py              # Interfaccia Gradio
├── config.py                # Dataset HF, percorsi, modello CLIP
├── scripts/
│   ├── download_dataset.py  # Fase 0: download da Hugging Face
│   └── precompute_embeddings.py  # Fase 1: embedding + FAISS
├── src/
│   ├── embedder.py          # CLIP (OpenCLIP ViT-B/32)
│   └── search.py            # Indice e ricerca
└── data/                    # Creato dagli script (non in git)
    ├── images/              # PNG 00000.png …
    ├── embeddings/
    └── index/
```

## Come funziona tecnicamente

1. **Embedding**: [OpenCLIP](https://github.com/mlfoundations/open_clip) `ViT-B-32` (pretrained OpenAI) → vettore 512-d, normalizzato L2.
2. **Indice**: [FAISS](https://github.com/facebookresearch/faiss) `IndexFlatIP` (prodotto scalare = coseno se i vettori sono normalizzati).
3. **Query**: stesso modello sulla foto utente → top-k vicini nell’indice.

## Scaricare / spostare la cartella

Puoi comprimere l’intera cartella del progetto (esclusa `.venv` e opzionalmente `data/` se lo rigeneri):

```bash
cd ~/Projects
zip -r image-similarity-prototype.zip image-similarity-prototype \
  -x "*/.venv/*" -x "*/data/images/*"
```

Su un altro Mac/PC: estrai, crea il venv, `pip install -r requirements.txt`, poi ripeti download + precompute (o copia anche `data/` se già pronto).

## Personalizzazione

- **Altro dataset HF**: cambia `HF_DATASET_ID` in `config.py` e adatta `download_dataset.py` se le colonne differiscono.
- **Modello più accurato**: in `config.py` prova `CLIP_MODEL = "ViT-L-14"` (più lento, più RAM).
- **Soglia minima**: in `app/main.py` filtra i risultati con `score < 0.25`.

## Riferimenti

- Karras et al., *A Style-Based Generator Architecture for GANs* — [FFHQ paper](https://arxiv.org/abs/1812.04948)
- Dataset HF: https://huggingface.co/datasets/nuwandaa/ffhq128
