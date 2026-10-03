"""4K smoke-test input (clearly labelled): upscale a 1080p clip to 3840x2160 and scale its calibration by 2.
This tests decoding, memory and runtime at 4K, NOT 4K image quality."""
import json, os, shutil, sys
import cv2
src_dir, dst_dir = sys.argv[1], sys.argv[2]
os.makedirs(dst_dir, exist_ok=True)
cap = cv2.VideoCapture(os.path.join(src_dir, "flight.mp4"))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
out = cv2.VideoWriter(os.path.join(dst_dir, "flight_4k.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (3840, 2160))
n = 0
while True:
    ok, img = cap.read()
    if not ok:
        break
    out.write(cv2.resize(img, (3840, 2160), interpolation=cv2.INTER_CUBIC)); n += 1
out.release()
cam = json.load(open(os.path.join(src_dir, "camera.json")))
K = cam["intrinsic_matrix"]
cam["intrinsic_matrix"] = [[K[0][0] * 2, 0, K[0][2] * 2], [0, K[1][1] * 2, K[1][2] * 2], [0, 0, 1]]
cam["width"], cam["height"] = 3840, 2160
cam["note"] = "4K smoke test: bicubic-upscaled from the 1080p derived video; distortion coefficients unchanged (normalised)."
json.dump(cam, open(os.path.join(dst_dir, "camera.json"), "w"), indent=2)
shutil.copy(os.path.join(src_dir, "telemetry_v2.csv"), os.path.join(dst_dir, "telemetry_v2.csv"))
print("frames", n, "fps", fps)
