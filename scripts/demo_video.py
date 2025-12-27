# scripts/demo_video.py - 비디오 파일에서 YOLOv8 객체 감지 데모
import cv2
import time
import json
from ultralytics import YOLO
from pathlib import Path

# 초기값 설정
prev_t = time.perf_counter()
fps = 0.0

root = Path(__file__).resolve().parents[1]
video = root / "samples" / "videos" / "sample.mp4"
screenshot_path = root / "assets" / "images" / "storeguard_d1_video_loop.jpg"

cap = cv2.VideoCapture(str(video))
if not cap.isOpened():
    raise SystemExit(f"failed to open: {video}")

# ROI 구성 로드
cfg_path = root / "config" / "roi.json"

data = json.loads(cfg_path.read_text(encoding="utf-8"))
rois = data["rois"]
roi = next(r for r in rois if r["id"] == "orange_zone_wide")  # or tight

# 원본 비디오 FPS 및 딜레이 계산
src_fps = cap.get(cv2.CAP_PROP_FPS)
if not src_fps or src_fps <= 1:
    src_fps = 25  # fallback
delay_ms = max(1, int(1000 / src_fps))
print(f"source fps={src_fps:.2f}, delay_ms={delay_ms}")


# YOLOv8 모델 로드
model = YOLO("yolov8n.pt")
cooldown_s = 2.0
last_intrusion_t = 0.0

# 비디오 루프 재생
while True:
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # FPS 계산
    now = time.perf_counter()
    dt = now - prev_t
    prev_t = now
    inst = 1.0 / dt if dt > 0 else 0.0
    fps = inst if fps == 0.0 else (0.9 * fps + 0.1 * inst)

    # YOLO 추론
    res = model.predict(
        frame,
        imgsz=320,
        conf=0.35,
        iou=0.50,
        max_det=5,
        classes=0,
        verbose=False,
    )[0]

    # ROI + intrusion
    x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
    intruded = False

    # ROI 표시(가시화)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)  # 주황 느낌

    for b in res.boxes:
        xA, yA, xB, yB = map(int, b.xyxy[0].tolist())
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

        cx = (xA + xB) // 2
        cy = int(yA + 0.2 * (yB - yA))

        inside = (x1 <= cx <= x2 and y1 <= cy <= y2)
        if inside:
            intruded = True
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        else:
            cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

    # FPS overlay
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    now_t = time.perf_counter()

    # intrusion 이벤트 발생 시각 갱신(쿨다운)
    if intruded and (now_t - last_intrusion_t) >= cooldown_s:
        last_intrusion_t = now_t
        print(f"[INTRUSION] roi={roi['id']} t={now_t:.2f}")

    # 최근 이벤트 후 0.8초 동안만 경고 표시
    if (now_t - last_intrusion_t) < 0.8:
        cv2.putText(frame, "INTRUSION!", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow("StoreGuard - Video Loop", frame)

    key = cv2.waitKey(delay_ms) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(screenshot_path), frame)
        print(f"saved: {screenshot_path}")

cap.release()
cv2.destroyAllWindows()
