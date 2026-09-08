#!/usr/bin/env bash
# One-command setup for the free CPU path.
#   ./scripts/setup.sh          # free path only
#   ./scripts/setup.sh --gpu    # also install the GPU model stack
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v ffmpeg >/dev/null; then
  echo "ffmpeg not found. Install it first:"
  echo "  Ubuntu/Debian : sudo apt-get install -y ffmpeg fonts-noto-telugu"
  echo "  macOS         : brew install ffmpeg"
  echo "  Windows       : winget install Gyan.FFmpeg"
  echo "(or rely on the bundled imageio-ffmpeg build, installed below)"
fi

python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt -r requirements-free.txt

if [[ "${1:-}" == "--gpu" ]]; then
  ./.venv/bin/pip install -r requirements-gpu.txt
fi

echo
echo "Done. Activate it and check your machine:"
echo "  source .venv/bin/activate"
echo "  export PYTHONPATH=src"
echo "  python -m telugu_dub doctor --config config/free_cpu.yaml --mode video"
