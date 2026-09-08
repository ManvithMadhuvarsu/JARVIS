#!/usr/bin/env bash
# Dub one clip in every mode you can run on this machine, into runs/demo_<mode>/.
#
#   ./scripts/run_all_modes.sh samples/source.mp4 [config] [transcript.srt]
#
# Audio and video always run. video-lipsync runs only if `doctor` says the GPU
# and the model checkout are there.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src

VIDEO="${1:?usage: run_all_modes.sh VIDEO [CONFIG] [TRANSCRIPT.srt]}"
CONFIG="${2:-config/free_cpu.yaml}"
TRANSCRIPT="${3:-}"

EXTRA=()
[[ -n "$TRANSCRIPT" ]] && EXTRA+=(--transcript "$TRANSCRIPT")

for MODE in audio video; do
  echo
  echo "════════ $MODE ════════"
  python -m telugu_dub run --video "$VIDEO" --config "$CONFIG" \
      --mode "$MODE" --voice preset --workdir "runs/demo_$MODE" "${EXTRA[@]}"
done

echo
echo "════════ video-lipsync ════════"
if python -m telugu_dub doctor --config "$CONFIG" --mode video-lipsync \
       --voice preset >/dev/null 2>&1; then
  python -m telugu_dub run --video "$VIDEO" --config "$CONFIG" \
      --mode video-lipsync --voice preset --workdir runs/demo_lipsync "${EXTRA[@]}"
else
  echo "skipped — this machine is not set up for lip-sync:"
  python -m telugu_dub doctor --config "$CONFIG" --mode video-lipsync \
      --voice preset || true
fi

echo
echo "Results:"
ls -1 runs/demo_*/final_te.* 2>/dev/null || true
