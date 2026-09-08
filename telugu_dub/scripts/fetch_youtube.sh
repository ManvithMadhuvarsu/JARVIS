#!/usr/bin/env bash
# Fetch a real sample clip (needs network access to YouTube).
#   ./scripts/fetch_youtube.sh "https://www.youtube.com/watch?v=XXXX" 30 60
set -euo pipefail
URL="${1:?usage: fetch_youtube.sh URL [START_SEC END_SEC]}"
START="${2:-}"; END="${3:-}"
OUT_DIR="samples"; mkdir -p "$OUT_DIR"
ARGS=(-o "$OUT_DIR/source.%(ext)s"
      -f 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]'
      --merge-output-format mp4)
if [[ -n "$START" && -n "$END" ]]; then
  ARGS+=(--download-sections "*${START}-${END}" --force-keyframes-at-cuts)
fi
# Official English captions make a free, exact substitute for the ASR stage.
yt-dlp "${ARGS[@]}" --write-auto-subs --write-subs --sub-langs "en.*" \
       --convert-subs srt "$URL"
echo "Downloaded to $OUT_DIR/"
