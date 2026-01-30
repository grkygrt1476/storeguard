# RTSP + DeepStream 8.0 Tracking Record Notes

This document is a reproducible debug log for:
- RTSP ingest (MediaMTX + FFmpeg publisher)
- `nvinfer` (primary detector) + `nvtracker`
- headless file recording (MKV/H264)
- evidence artifacts (video + frames + logs)

## TL;DR (What worked)
- Use file source baseline first (remove RTSP variables).
- Keep all model/engine paths under `/workspace/storeguard/...` (avoid `/opt/...` permission issues for non-root).
- `batch-size=1` for single source (engine filename must match).
- Use graceful quit: `(sleep 20; echo q) | deepstream-app ...` to flush muxer (avoid 0B files).

## Environment
- Host: WSL2 Ubuntu
- Docker image: `nvcr.io/nvidia/deepstream:8.0-samples-multiarch`
- Run style (non-root + bind mount):
  - host `~/storeguard` -> container `/workspace/storeguard`
  - `--user $(id -u):$(id -g)`
  - `--network ds-rtsp`

## Evidence (expected outputs)
- Video: `outputs/videos/_verify_tracker_overlay_check.mkv` (non-0B, playable)
- Frames:
  - `outputs/videos/_verify_frame0.png`
  - `outputs/videos/_verify_frame60.png`
  - `outputs/videos/_verify_frame120.png`
- Log (example): `outputs/logs/verify_tracker_overlay_<timestamp>.log`

## Step 1) Copy baseline configs from DeepStream container
- Copy these files into repo `config/`:
  - `source4_1080p_dec_infer-resnet_tracker_sgie_tiled_display.txt`
  - `config_infer_primary.txt`
- Rename to:
  - `config/_baseline_ds_tracker_record.txt`
  - `config/_baseline_infer_primary.txt`

## Step 2) Baseline app config edits (file source + headless record)
File: `config/_baseline_ds_tracker_record.txt`
- Source:
  - `type=2` (URI)
  - `uri=file:///workspace/storeguard/assets/videos/sample_1080p_h265.mp4`
  - `num-sources=1`
- Disable tiled display (headless):
  - `[tiled-display] enable=0`
- File sink:
  - `[sink0] type=3`
  - `container=2` (MKV)
  - `codec=1` (H264)
  - `enc-type=0` (HW encoder)
  - `output-file=outputs/videos/_verify_tracker_overlay_check.mkv`
- Single-source batch:
  - `[streammux] batch-size=1`
  - `[primary-gie] batch-size=1`
- Primary GIE config path:
  - `[primary-gie] config-file=/workspace/storeguard/config/_baseline_infer_primary.txt`
- Tracker config path:
  - use absolute path to `config_tracker_NvDCF_perf.yml`
- Disable secondary GIEs:
  - `[secondary-gie0] enable=0`
  - `[secondary-gie1] enable=0`

## Step 3) Baseline infer config edits (all paths under /workspace)
File: `config/_baseline_infer_primary.txt`
- Under `[property]`:
  - `onnx-file=/workspace/storeguard/models/Primary_Detector/resnet18_trafficcamnet_pruned.onnx`
  - `model-engine-file=/workspace/storeguard/models/Primary_Detector/resnet18_trafficcamnet_pruned.onnx_b1_gpu0_fp16.engine`
  - `labelfile-path=/workspace/storeguard/models/Primary_Detector/labels.txt`
  - `batch-size=1`

## Step 4) Model prep (copy ONNX/labels into workspace)
```bash
docker run --rm --gpus all \
  -v ~/storeguard:/workspace/storeguard \
  nvcr.io/nvidia/deepstream:8.0-samples-multiarch bash -lc \
  "mkdir -p /workspace/storeguard/models/Primary_Detector && \
   cp -f /opt/nvidia/deepstream/deepstream-8.0/samples/models/Primary_Detector/resnet18_trafficcamnet_pruned.onnx /workspace/storeguard/models/Primary_Detector/ && \
   cp -f /opt/nvidia/deepstream/deepstream-8.0/samples/models/Primary_Detector/labels.txt /workspace/storeguard/models/Primary_Detector/"
```

## Step 5) Run baseline (graceful quit) + verify
```bash
TAG=verify_tracker_overlay
TS=$(date +%Y%m%d_%H%M%S)
LOG=outputs/logs/${TAG}_${TS}.log

docker run --rm --gpus all --user $(id -u):$(id -g) \
  -v ~/storeguard:/workspace/storeguard -w /workspace/storeguard \
  nvcr.io/nvidia/deepstream:8.0-samples-multiarch bash -lc \
  "(sleep 20; echo q) | deepstream-app -c config/_baseline_ds_tracker_record.txt 2>&1 | tee $LOG || true"

grep -Ei "deserialize engine|serialize cuda engine|failed to serialize|NvMultiObjectTracker" "$LOG" | tail -n 120
ls -lh outputs/videos | rg "_verify_tracker_overlay_check|_verify_frame" || true
```

## step 6) Extract 3 frames (0/60/120)
```bash
ffmpeg -hide_banner -loglevel error -i outputs/videos/_verify_tracker_overlay_check.mkv \
  -vf "select='eq(n,0)'" -vframes 1 outputs/videos/_verify_frame0.png

ffmpeg -hide_banner -loglevel error -i outputs/videos/_verify_tracker_overlay_check.mkv \
  -vf "select='eq(n,60)'" -vframes 1 outputs/videos/_verify_frame60.png

ffmpeg -hide_banner -loglevel error -i outputs/videos/_verify_tracker_overlay_check.mkv \
  -vf "select='eq(n,120)'" -vframes 1 outputs/videos/_verify_frame120.png
```