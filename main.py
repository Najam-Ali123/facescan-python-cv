"""
Real-time Face, Age & Gender Detection
---------------------------------------
Opens your webcam, detects any face(s) in the frame, and overlays an
estimated age range and gender for each one.

Models used (all loaded through cv2.dnn, no training required):
  - Face detection : OpenCV's pretrained SSD (ResNet-10 backbone)
  - Age prediction  : Caffe CNN trained by Levi & Hassner, 8 age buckets
  - Gender prediction: Caffe CNN trained by Levi & Hassner, Male/Female

Controls:
  q   quit
  s   save a snapshot of the current frame to snapshots/
"""
import argparse
import os
import time

import cv2 as cv
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")

FACE_PROTO = os.path.join(MODELS_DIR, "opencv_face_detector.pbtxt")
FACE_MODEL = os.path.join(MODELS_DIR, "opencv_face_detector_uint8.pb")
AGE_PROTO = os.path.join(MODELS_DIR, "age_deploy.prototxt")
AGE_MODEL = os.path.join(MODELS_DIR, "age_net.caffemodel")
GENDER_PROTO = os.path.join(MODELS_DIR, "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(MODELS_DIR, "gender_net.caffemodel")
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_BUCKETS = [
    "(0-2)", "(4-6)", "(8-12)", "(15-20)",
    "(25-32)", "(38-43)", "(48-53)", "(60-100)",
]
GENDER_LIST = ["Male", "Female"]
FACE_CONF_THRESHOLD = 0.7
PADDING = 20 
def check_models_exist():
    required = [FACE_PROTO, FACE_MODEL, AGE_PROTO, AGE_MODEL, GENDER_PROTO, GENDER_MODEL]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print("Missing model file(s):")
        for f in missing:
            print(f"  - {f}")
        print("\nRun `python download_models.py` first (see README.md).")
        raise SystemExit(1)


def load_networks():
    face_net = cv.dnn.readNet(FACE_MODEL, FACE_PROTO)
    age_net = cv.dnn.readNet(AGE_MODEL, AGE_PROTO)
    gender_net = cv.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
    return face_net, age_net, gender_net


def detect_faces(net, frame, conf_threshold=FACE_CONF_THRESHOLD):
    """Runs the SSD face detector and returns a list of (x1, y1, x2, y2) boxes."""
    h, w = frame.shape[:2]
    blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            boxes.append((max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)))
    return boxes
def predict_age_gender(age_net, gender_net, face_img):
    blob = cv.dnn.blobFromImage(
        face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False
    )
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]
    gender_confidence = float(gender_preds[0].max())

    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_BUCKETS[age_preds[0].argmax()]
    age_confidence = float(age_preds[0].max())

    return gender, gender_confidence, age, age_confidence

def run(camera_index=0):
    check_models_exist()
    print("Loading models...")
    face_net, age_net, gender_net = load_networks()

    cap = cv.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam at index {camera_index}. "
            "Try a different --camera index, or check OS camera permissions."
        )
    print("Webcam started. Press 'q' to quit, 's' to save a snapshot.")
    prev_time = time.time()
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to grab frame, exiting.")
            break
        frame = cv.flip(frame, 1) 
        boxes = detect_faces(face_net, frame)

        for (x1, y1, x2, y2) in boxes:
            # add padding around the face before cropping, clamped to frame bounds
            fy1 = max(0, y1 - PADDING)
            fy2 = min(frame.shape[0] - 1, y2 + PADDING)
            fx1 = max(0, x1 - PADDING)
            fx2 = min(frame.shape[1] - 1, x2 + PADDING)
            face_img = frame[fy1:fy2, fx1:fx2]

            if face_img.size == 0:
                continue

            gender, gconf, age, aconf = predict_age_gender(age_net, gender_net, face_img)

            label = f"{gender} ({gconf*100:.0f}%), {age} ({aconf*100:.0f}%)"

            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
            cv.putText(
                frame, label, (x1, label_y),
                cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv.LINE_AA,
            )
        now = time.time()
        fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
        prev_time = now
        cv.putText(
            frame, f"FPS: {fps:.1f}", (10, 25),
            cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv.LINE_AA,
        )

        cv.imshow("Face, Age & Gender Detection - press q to quit", frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = os.path.join(SNAPSHOT_DIR, f"snapshot_{int(time.time())}.png")
            cv.imwrite(fname, frame)
            print(f"Saved {fname}")

    cap.release()
    cv.destroyAllWindows()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time face, age & gender detection")
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Webcam device index (default: 0). Try 1, 2... if you have multiple cameras.",
    )
    args = parser.parse_args()
    run(camera_index=args.camera)
