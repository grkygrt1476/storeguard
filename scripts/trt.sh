#!/usr/bin/env bash
set -euo pipefail

IMAGE="${TRT_IMAGE:-nvcr.io/nvidia/tensorrt:25.12-py3}"
docker run --rm --gpus all \
  -v "$(pwd)":/workspace/storeguard \
  -w /workspace/storeguard \
  "$IMAGE" "$@"
