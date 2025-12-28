# storeguard TensorRT helpers (loaded only when venv is activated inside this repo)

trt() {
  docker run --rm --gpus all \
    -v "$PWD":/workspace/storeguard -w /workspace/storeguard \
    nvcr.io/nvidia/tensorrt:25.12-py3 \
    "$@"
}

trtexec_trt() {
  trt trtexec "$@"
}
