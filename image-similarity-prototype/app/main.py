#!/usr/bin/env python3
"""
Fase 2–3: interfaccia Gradio — carica una foto, confronta embedding, mostra i match.

Uso:
  python app/main.py
  python app/main.py --share   # link pubblico temporaneo Gradio
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr
from PIL import Image

from config import TOP_K_DEFAULT
from src.embedder import ClipEmbedder
from src.search import SimilaritySearch


def build_ui(search: SimilaritySearch, embedder: ClipEmbedder):
    def query_image(user_image: Image.Image, top_k: int):
        if user_image is None:
            return None, "Carica un'immagine per iniziare."

        query_vec = embedder.encode_pil(user_image)
        hits = search.search(query_vec, top_k=int(top_k))

        gallery_paths = []
        lines = []
        for rank, (img_id, score) in enumerate(hits, start=1):
            path = Path(img_id)
            if path.exists():
                gallery_paths.append(str(path))
            lines.append(f"{rank}. {path.name} — similarità {score:.4f}")

        caption = "\n".join(lines) if lines else "Nessun risultato."
        return gallery_paths, caption

    with gr.Blocks(title="Ricerca per similarità visiva") as demo:
        gr.Markdown(
            """
            # Prototipo: ricerca per similarità (FFHQ 128×128)

            1. **Offline**: embedding precalcolati sul dataset (`precompute_embeddings.py`)
            2. **Online**: embedding della tua foto → confronto con FAISS (similarità coseno)
            3. **Risultato**: immagini più simili del catalogo
            """
        )
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="La tua foto")
                top_k = gr.Slider(1, 24, value=TOP_K_DEFAULT, step=1, label="Numero risultati")
                btn = gr.Button("Cerca simili", variant="primary")
            with gr.Column():
                gallery = gr.Gallery(label="Immagini più simili", columns=4, height=400)
                info = gr.Textbox(label="Dettaglio", lines=12)

        btn.click(fn=query_image, inputs=[input_image, top_k], outputs=[gallery, info])
        input_image.change(fn=query_image, inputs=[input_image, top_k], outputs=[gallery, info])

    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Crea link pubblico Gradio")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()

    print("Caricamento indice e modello CLIP…")
    search = SimilaritySearch()
    embedder = ClipEmbedder()
    demo = build_ui(search, embedder)
    demo.launch(server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
