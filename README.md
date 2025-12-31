# storeguard
Real-time unmanned store CCTV anomaly detection (ONNX + TensorRT).무인매장 CCTV 영상을 처리하고 이상행동 감지 프로젝트

무인가게(또는 CCTV 환경)에서 발생할 수 있는 이상행동을 **영상 파이프라인**으로 탐지/표시하는 미니 데모입니다.  
5일 안에 “돌아가는 증거”를 남기는 것이 목표이며, 이후 상용 최적화 루트(ONNX Runtime/TensorRT)까지 확장 가능한 구조로 설계합니다.

## architecture
```mermaid
---
config:
  theme: default
---
flowchart TB
classDef io stroke-width:1px;
classDef proc stroke-width:1px;
classDef srv stroke-width:1px,stroke-dasharray: 3 3;
classDef note stroke-width:1px,stroke-dasharray: 2 2;

subgraph D4["D4 — E2E Triton pipeline (evidence-backed)"]
direction TB

  V["Video Loop<br/>(read frames)"]:::io
  Dp["Real-time Frame Skipping (lag-based)<br/>buffer=2 / keep=latest"]:::proc
  P["Preprocess<br/>resize + normalize<br/>→ 1×3×320×320 (NCHW)"]:::proc

  TC["Triton Client<br/>(HTTP)"]:::proc
  TS["Triton Server<br/>(ONNXRuntime GPU)<br/>YOLOv8n_320"]:::srv

  Y["Model Output<br/>1×84×2100"]:::io
  N["Postprocess<br/>decode + NMS"]:::proc
  R["Detections<br/>(boxes / scores / classes)"]:::io

  Evt["Alert Decision<br/>ROI hit ratio + loiter_sec"]:::proc

  O["Overlay + Encode<br/>(write video / stream)"]:::proc
  M["Metrics / Logs<br/>drops, fps, p95(total/e2e)"]:::note

  V --> Dp --> P --> TC --> TS --> Y --> N --> R --> Evt --> O --> M
end

subgraph D5["D5 — Event-gated Classifier (ONNX)"]
direction TB
  Samp["Clip sampler<br/>(ROI/person crop → T frames)"]:::proc
  CIn["Clip input<br/>(1×T×3×224×224)"]:::io
  CL["ONNXRuntime<br/>(CUDA EP)"]:::proc
  COut["Logit<br/>→ score / flag"]:::io
  Cm["CLS Metrics<br/>cls_calls, latency.cls"]:::note

  Samp --> CIn --> CL --> COut --> Cm
end

Evt -. "event active (intrusion/loitering)" .-> Samp
```


## Goals
- 영상 입력(loop) → 추론 → 오버레이 출력까지 **엔드투엔드 파이프라인 증명**
- baseline vs 최적화(ONNX Runtime → TensorRT)로 **FPS/latency 비교 지표** 남기기
- 결과물: 실행 방법, 스크린샷, 성능 로그, 간단한 아키텍처/설명(README)

## Scope (5 days)
- D1: Video loop + FPS overlay + screenshot
- D2: ROI intrusion event (YOLO person + head-point rule)
- D3: TensorRT FP16 engine build + evidence log + benchmark
- D4: Triton(ONNXRuntime GPU) serving + E2E FPS/p95 측정
- D5: Clip classifier wiring (ONNXRuntime CUDA) + metrics(cls_calls/latency.cls) + packaging

## Demo (D1-2 Video Loop MVP)
```bash
python scripts/demo_video.py
```
- q: quit
- s: save screenshot to assets/images/storeguard_d1_video_loop.jpg

![D1 video loop](assets/images/storeguard_d1_video_loop.jpg)

## Demo (D2 Intrusion Event)
```bash
python scripts/demo_video.py
```
- q: quit
- s: save screenshot (e.g., assets/images/storeguard_d2_intrusion.jpg)

Intrusion rule: if "head-point" (bbox center at 20% height) enters ROI rectangle, trigger event with cooldown.
![D2 intrusion](assets/images/storeguard_d2_intrusion.jpg)

## ONNX Smoke Test (WSL GPU)
```bash
./scripts/run_onnx.sh scripts/onnx_smoke.py
./scripts/run_onnx.sh -c "import onnxruntime as ort; print(ort.get_available_providers())"
```
## NGC (nvcr.io) Login (for TensorRT container)
- Username: `$oauthtoken`
- Password: NGC API Key 

## TensorRT FP16 (D3)

We export YOLOv8n to ONNX and build a TensorRT FP16 engine using `trtexec` (via NGC Docker image).
This leaves reproducible evidence logs and produces an `.engine` for future Triton serving.
- Input ONNX: `outputs/models/yolov8n_320.onnx`
- Output Engine: `outputs/engine/yolov8n_320_fp16.engine`
- Evidence log: `outputs/logs/d3_trt_build_fp16.log`
- Benchmark (trtexec summary): Throughput ~742 qps, mean latency ~1.27 ms (RTX 4080 SUPER, FP16)

### Performance (single-batch, 320x320, RTX 4080 SUPER)
| Runtime | Precision | Mean Latency (ms) | Throughput (qps) | Evidence |
|---|---:|---:|---:|---|
| TensorRT | FP16 | 1.27 | 741.8 | outputs/logs/d3_trt_build_fp16.log |

### 1) Run TensorRT container (WSL/GPU)
We use the NGC TensorRT container so local TensorRT installation is not required.

### 2) Build FP16 engine (ONNX → .engine)
```bash
mkdir -p outputs/engine outputs/logs

trtexec_trt \
  --onnx=outputs/models/yolov8n_320.onnx \
  --saveEngine=outputs/engine/yolov8n_320_fp16.engine \
  --fp16 \
  2>&1 | tee outputs/logs/d3_trt_build_fp16.log
```
### Environment stamp (D3)
- Host: RTX 4080 SUPER (WSL2), see `nvidia-smi` in `outputs/logs/d3_trt_stamp.log`
- Docker: see `outputs/logs/d3_trt_stamp.log`
- TensorRT container: `nvcr.io/nvidia/tensorrt:25.12-py3` (TensorRT 10.14.1)

### Reproducible TensorRT commands (no shell functions)
- `scripts/trt.sh`: runs commands inside the TensorRT NGC container
- `scripts/trtexec_trt.sh`: convenience wrapper for `trtexec`

Example:
```bash
./scripts/trtexec_trt.sh --help | head
```

## 15 완료 판정
- 새 셸(함수 없는 상태)에서 아래가 되면 끝
```bash
# 1) trtexec 동작 확인
./scripts/trtexec_trt.sh -h | head

# 2) TensorRT 버전 스탬프
./scripts/trt.sh python -c "import tensorrt as trt; print(trt.__version__)"

```

## Triton Quickstart (D4 / P0: ONNX Serving)

Serve the exported ONNX model with Triton and run a 1-shot HTTP inference.

### 0) Prereq
- Docker + NVIDIA Container Toolkit (GPU)
- `docker login nvcr.io` (NGC)

### 1) Start Triton server (ONNX model repo)
```bash
mkdir -p outputs/logs

docker run --rm --gpus all --net=host \
  -v "$PWD/models":/models \
  nvcr.io/nvidia/tritonserver:25.12-py3 \
  tritonserver --model-repository=/models \
  2>&1 | tee outputs/logs/d4_triton_server_onnx.log
```
### 2) Check model is ready
```bash
curl -s localhost:8000/v2/health/ready && echo
curl -s localhost:8000/v2/models/yolov8n_320_onnx && echo
```
### 3) 1-shot inference (client)
```bash
pip install -q "tritonclient[http]" numpy
python scripts/triton_client_once.py 2>&1 | tee outputs/logs/d4_triton_client_once.log
```
## Triton Serving (D3) — TensorRT Engine (P1 Stamp)

We serve the TensorRT FP16 engine using NVIDIA Triton Inference Server and leave reproducible server/client logs.

- Model: `yolov8n_320_trt` (`platform: tensorrt_plan`)
- Input: `images` = `[1,3,320,320]` FP32
- Output: `output0` = `[1,84,2100]` FP32
- Evidence logs:
  - Server: `outputs/logs/d3_triton_server_trt.log`
  - Client: `outputs/logs/d3_triton_client_once_trt.log`

### 1) Start Triton server (GPU)
```bash
mkdir -p outputs/logs

docker run --rm --gpus all \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "$PWD/models":/models \
  --name triton_storeguard \
  nvcr.io/nvidia/tritonserver:25.12-py3 \
  tritonserver --model-repository=/models \
  2>&1 | tee outputs/logs/d3_triton_server_trt.log
```
### 2) Health + model metadata
```bash
curl -sS -o /dev/null -w "ready http=%{http_code}\n" http://127.0.0.1:8000/v2/health/ready
curl -sS http://127.0.0.1:8000/v2/models/yolov8n_320_trt | head
```
### 3) 1-shot inference (client)
```bash
python scripts/triton_client_once.py 2>&1 | tee outputs/logs/d3_triton_client_once_trt.log
```

## Modeling Track (v1) — Clip-sampled Video Classifier (MobileNetV3 + TAP)

This branch adds a lightweight **video-level classifier** to complement the real-time detector pipeline.  
Goal: **fast, deployable, and time-aware** modeling with reproducible evidence (logs/plots).

---

### Data
- Source: Kaggle shoplifting videos (≈10s each).
- Labels: `normal` vs `shoplifting`
- Raw videos are **NOT** committed (scripts/CSV/metrics only).

---

### Method
- **Backbone**: MobileNetV3 (ImageNet pretrained) as per-frame feature extractor
- **Temporal module**: Temporal Attention Pooling (TAP) over `T=16` frames
- **Head**: 1-layer binary classifier
- **Video-level scoring**: `top-k mean` aggregation over multiple sampled clips per video (default `k=3`)
- **Training**: transfer learning (freeze backbone for first N epochs, then unfreeze), AMP enabled  
  - Note: Freezing reduces **backprop** cost, but **decode/preprocess + forward** still dominate, so epoch time may remain similar (cache helps most).

---

### Pipeline
1) `videos.csv` → generate train/val splits  
2) sample clip windows → `data/clips/*_clips.csv`  
3) (optional but recommended) cache decoded frames into `.npz` to remove decode bottleneck  
4) train with early stopping + export metrics / best checkpoint  
5) plot curves from `metrics.json`

---

### Reproduce
```bash
# 1) generate clip sampling CSVs
python scripts/modeling/make_clips.py

# 2) optional cache (faster epochs)
python scripts/modeling/cache_video_npz.py --cache_fps 8 --size 256

# 3) train (example: seed sweep)
BASE=outputs/modeling/run_006_seed_sweep
for S in 42 43 44; do
  python scripts/modeling/train_v1.py \
    --use_cache --epochs 30 --early_patience 5 \
    --batch 8 --amp --num_workers 8 \
    --dropout 0.3 --wd 3e-4 --video_topk 3 \
    --seed $S --out_dir ${BASE}_seed${S}
done

# 4) plot metrics (example: headline run)
python scripts/modeling/plot_metrics.py \
  --metrics outputs/modeling/run_006_seed_sweep_seed43/metrics.json \
  --out outputs/modeling/run_006_seed_sweep_seed43/history.png

```
## Results (video-level, top-k mean)

### Results (video-level, top-k mean)

| Run | Seed | Best `video_acc_topkmean` |
|---|---:|---:|
| `run_006_seed_sweep_seed42` | 42 | 0.9167 |
| `run_006_seed_sweep_seed43` | 43 | 0.9444 |
| `run_006_seed_sweep_seed44` | 44 | 0.9444 |

**Recommendation:** Use **one best run** as the headline (plot + `metrics.json`), and keep the seed sweep as stability evidence.

---

### Evidence
- Loader/decoder checks: `outputs/logs/loader_smoke_*.txt`
- Clip decode sample: `outputs/videos/smoke_clip_*.mp4`
- Training metrics: `outputs/modeling/run_*/metrics.json`
- Curves: `outputs/modeling/run_*/plots/history_*.png`

---

### Notes / Limitations
- Dataset is small and may not match Korean store CCTV domain.


### Headline run plots
Run: `run_006_seed_sweep_seed43`

<p align="center">
  <img src="outputs/modeling/run_006_seed_sweep_seed43/plots/history_metrics.png" width="720" />
</p>

<p align="center">
  <img src="outputs/modeling/run_006_seed_sweep_seed43/plots/history_loss.png" width="720" />
</p>

<p align="center">
  <img src="outputs/modeling/run_006_seed_sweep_seed43/plots/history_time.png" width="720" />
</p>

### Next
- Replace dataset (AIHub/partner data)
- Add temporal head (tiny transformer)
- Integrate into serving pipeline

## Ablation v1 — Tiny Temporal Transformer Head (TT)

TAP(Temporal Attention Pooling) 대신 **Tiny Temporal Transformer**로 시간축 모델링을 교체한 실험입니다.  
Backbone(MobileNetV3) + 학습/데이터 파이프라인은 동일하고, **temporal head만 변경**했습니다.

### What changed
- `--temporal tap` → `--temporal tt`
- TT config (tiny):
  - `tt_d_model=64`, `tt_heads=2`, `tt_layers=1`, `tt_dropout=0.2`
- Cache `.npz` 로딩 시 key 호환: `cache_fps/native_fps` 사용 + (없으면 mp4 decode fallback)

### Reproduce (tiny TT seed sweep)
```bash
BASE=outputs/modeling/run_010_tt_tiny_seed_sweep
for S in 42 43 44; do
  python scripts/modeling/train_v2_tt.py \
    --temporal tt --use_cache --amp \
    --epochs 30 --early_patience 5 \
    --batch 8 --num_workers 8 \
    --dropout 0.4 --wd 5e-4 --video_topk 3 \
    --tt_d_model 64 --tt_heads 2 --tt_layers 1 --tt_dropout 0.2 \
    --seed $S \
    --out_dir ${BASE}_seed${S}
done

```
### Results (video-level, top-k mean)

| Run | Seed | Best `video_acc_topkmean` |
|---|---:|---:|
| `run_010_tt_tiny_seed_sweep_seed42` | 42 | 0.9167 |
| `run_010_tt_tiny_seed_sweep_seed43` | 43 | 0.9444 |
| `run_010_tt_tiny_seed_sweep_seed44` | 44 | 0.9167 |

**Recommendation:** Use `seed43` as the headline run, keep the sweep as stability evidence.

### Evidence
- Metrics: `outputs/modeling/run_010_tt_tiny_seed_sweep_seed*/metrics.json`
- Curves: `outputs/modeling/run_010_tt_tiny_seed_sweep_seed*/plots/history_*.png`

### Notes
- Dataset is small; performance is likely capped by data quantity/variety rather than the temporal head choice.

#### Curves (headline run: seed43)

![TT Val Metrics](outputs/modeling/run_010_tt_tiny_seed_sweep_seed43/plots/history_metrics.png)

<!-- optional -->
![Train Loss](outputs/modeling/run_010_tt_tiny_seed_sweep_seed43/plots/history_loss.png)

## D4 — E2E Triton pipeline (evidence-backed)

- Triton Inference Server로 **YOLOv8n ONNX** 모델을 서빙하고,
- 영상 프레임을 루프 처리하며 **preprocess → Triton HTTP inference → postprocess(decode+NMS) → ROI 이벤트 판정 → overlay+encode**까지
  end-to-end 데모를 구성했습니다.
- 결과물로 **output mp4 + metrics json**을 남기며, metrics에는 `drops`, `fps_effective`,
  `latency_ms(preprocess/infer/postprocess/encode/total)` 및 `events(intrusion_frames, loitering_frames)`가 기록됩니다.

## D5 — Classifier wiring (planned → implemented)

- D4의 이벤트 결과 뒤에 **clip-sampled video classifier(ONNXRuntime, GPU)** 를 “후단 refine”으로 연결했습니다.
- 이벤트가 활성화되면 ROI(or person) crop에서 **T=16 프레임 clip**을 구성해 classifier를 실행하고 score(logit)를 얻습니다.
- “실제로 돌았는지”를 증명하기 위해 metrics에 아래를 추가했습니다.
  - `cls_calls`: classifier 호출 횟수
  - `latency_ms.cls`: classifier latency 통계(mean/p95/max)

### Evidence
- Metrics JSON: `outputs/logs/*.json`