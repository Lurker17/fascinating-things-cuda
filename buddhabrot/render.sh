#!/usr/bin/env bash
# Build and render the looping video.
#   Usage: ./render.sh [W H seconds fps log2samples]
#   e.g.   ./render.sh                      -> 1920x1080, 20 s, 30 fps
#          ./render.sh 1920 1080 20 60      -> same loop, silky 60 fps
#          ./render.sh 3840 2160 20 30 29   -> 4K, more samples
set -euo pipefail
W=${1:-1920}; H=${2:-1080}; SECONDS_=${3:-20}; FPS=${4:-30}; LOG2S=${5:-28}
FRAMES=$((SECONDS_ * FPS))
nvcc -O3 -arch=sm_80 -o buddhabrot buddhabrot.cu
./buddhabrot "$W" "$H" "$FRAMES" 0 "$FRAMES" "$LOG2S" |
  ffmpeg -loglevel error -y -f rawvideo -pix_fmt rgb24 -s "${W}x${H}" -r "$FPS" -i - \
         -c:v libx264 -preset medium -crf 16 -pix_fmt yuv420p -movflags +faststart \
         divine_buddhabrot.mp4
echo "wrote divine_buddhabrot.mp4"
