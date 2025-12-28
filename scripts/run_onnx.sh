#!/usr/bin/env bash
set -euo pipefail

# WSL이면 WSL GPU lib 우선
if [ -d /usr/lib/wsl/lib ]; then
  export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}
fi

exec python "$@"

# chmod +x scripts/run_onnx.sh

# ./scripts/run_onnx.sh scripts/onnx_smoke.py
# ./scripts/run_onnx.sh -c "import onnxruntime as ort; print(ort.get_available_providers())"

