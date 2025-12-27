import cv2
import time
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

# 원본 비디오 FPS 및 딜레이 계산
src_fps = cap.get(cv2.CAP_PROP_FPS)
if not src_fps or src_fps <= 1:
    src_fps = 25  # fallback
delay_ms = max(1, int(1000 / src_fps))
print(f"source fps={src_fps:.2f}, delay_ms={delay_ms}")

# 비디오 루프 재생
while True:
    ok, frame = cap.read()

    # FPS 계산
    now = time.perf_counter()
    dt = now - prev_t
    prev_t = now

    inst = 1.0 / dt if dt > 0 else 0.0
    fps = inst if fps == 0.0 else (0.9 * fps + 0.1 * inst)  # 살짝 부드럽게

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    # 비디오 프레임 읽기
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

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
