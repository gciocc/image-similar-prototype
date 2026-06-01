#!/usr/bin/env bash
# Setup rapido del prototipo (macOS / Linux)
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

echo ""
echo "Prossimi passi:"
echo "  python scripts/download_dataset.py --max-images 500"
echo "  python scripts/precompute_embeddings.py"
echo "  python app/main.py"
