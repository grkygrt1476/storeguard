import cv2
from pathlib import Path

root = Path(__file__).resolve().parents[1]
video = root / "samples" / "videos" / "sample.mp4"

print("__file__ =", __file__)
print("resolved =", Path(__file__).resolve())
print("parents0 =", Path(__file__).resolve().parents[0])
print("parents1 =", Path(__file__).resolve().parents[1])

print("__file__=",__file__)
print("resolved=",Path(__file__).resolve())
print("parents0=",Path(__file__).resolve().parents[0])
print("parents1=",Path(__file__).resolve().parents[1])

cap = cv2.VideoCapture(str(video))
if not cap.isOpened():
    raise SystemExit(f"failed to open: {video}")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow("StoreGuard - Video Loop", frame)
    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
