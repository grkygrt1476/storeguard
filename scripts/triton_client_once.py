import numpy as np
import tritonclient.http as httpclient

URL = "localhost:8000"
MODEL = "yolov8n_320_onnx"

def main():
    client = httpclient.InferenceServerClient(url=URL, verbose=False)

    # YOLOv8n ONNX input: images [1,3,320,320] FP32
    x = np.random.rand(1, 3, 320, 320).astype(np.float32)

    inp = httpclient.InferInput("images", x.shape, "FP32")
    inp.set_data_from_numpy(x)

    out = httpclient.InferRequestedOutput("output0")

    res = client.infer(model_name=MODEL, inputs=[inp], outputs=[out])
    y = res.as_numpy("output0")

    print("OK infer")
    print("input:", x.shape, x.dtype)
    print("output:", y.shape, y.dtype, "min/max:", float(y.min()), float(y.max()))

if __name__ == "__main__":
    main()
