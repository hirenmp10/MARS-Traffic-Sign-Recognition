#!/usr/bin/env python3
import os
import argparse
import time
from collections import deque

import cv2
import numpy as np

from preprocess import preprocess_image, detect_sign_regions

# Force TensorFlow to run on CPU if GPU is not desired or unavailable
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
try:
    import tensorflow as tf
except ImportError as e:
    raise ImportError(
        'TensorFlow is required for this script.\n'
        'Install it with: python -m pip install --no-cache-dir -r requirements.txt\n'
        'If your C: drive is low on space, set TMP and TEMP to a larger drive before installing.'
    ) from e

MODEL_PATH = 'traffic_sign_model_clean.keras'

CLASS_LABELS = [
    "Speed 20","Speed 30","Speed 50","Speed 60","Speed 70",
    "Speed 80","End 80","Speed 100","Speed 120","No passing",
    "No passing >3.5t","Right of way","Priority road","Yield",
    "Stop","No vehicles","No trucks","No entry","Caution",
    "Curve left","Curve right","Double curve","Bumpy road",
    "Slippery","Narrows right","Road work","Traffic signals",
    "Pedestrians","Children","Bicycles","Ice/snow","Animals",
    "End restrictions","Turn right","Turn left","Straight",
    "Straight or right","Straight or left","Keep right",
    "Keep left","Roundabout","End no passing","End no passing >3.5t"
]


def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Model file not found: {path}')
    return tf.keras.models.load_model(path, compile=False)


def create_dashboard(panel_width, frame_height):
    return np.zeros((frame_height, panel_width, 3), dtype=np.uint8)


def draw_panel(panel, last_sign, history, count, confidence_log):
    cv2.putText(panel, 'TRAFFIC SIGN DASHBOARD', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.line(panel, (0, 45), (panel.shape[1], 45), (100, 100, 100), 1)

    cv2.putText(panel, 'Last Detected:', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(panel, last_sign[:36], (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    cv2.putText(panel, f'Total Signs: {count}', (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.line(panel, (0, 145), (panel.shape[1], 145), (100, 100, 100), 1)

    cv2.putText(panel, 'Recent History:', (10, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    for i, entry in enumerate(history):
        cv2.putText(panel, entry[:36], (10, 190 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    if confidence_log:
        recent = list(confidence_log)[-10:]
        bar_y = panel.shape[0] - 90
        cv2.putText(panel, 'Confidence:', (10, bar_y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
        for i, val in enumerate(recent):
            bar_h = int(val * 60)
            color = (0, 255, 0) if val > 0.85 else (0, 165, 255)
            x0 = 10 + i * 32
            x1 = x0 + 22
            y0 = bar_y + 60 - bar_h
            y1 = bar_y + 60
            cv2.rectangle(panel, (x0, y0), (x1, y1), color, -1)
            cv2.putText(panel, f'{val:.2f}', (x0, y1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)


def predict_sign(model, roi):
    processed = preprocess_image(roi)
    if processed is None:
        return None, 0.0

    preds = model.predict(processed, verbose=0)
    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds))
    return CLASS_LABELS[class_id], confidence


def main():
    parser = argparse.ArgumentParser(description='Run traffic sign dashboard without ROS or PyTorch')
    parser.add_argument('--source', default='0', help='Camera index or video file path')
    parser.add_argument('--threshold', type=float, default=0.55, help='Prediction confidence threshold')
    parser.add_argument('--panel-width', type=int, default=360, help='Dashboard panel width')
    args = parser.parse_args()

    source = args.source
    if source.isdigit():
        source = int(source)

    model = load_model(MODEL_PATH)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f'Unable to open video source: {args.source}')

    last_sign = 'Waiting...'
    history = deque(maxlen=6)
    confidence_log = deque(maxlen=50)
    count = 0

    cv2.namedWindow('Traffic Sign Dashboard', cv2.WINDOW_NORMAL)

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = detect_sign_regions(frame)
        for x, y, w, h, area in boxes:
            if area < 800:
                continue

            roi = frame[y:y+h, x:x+w]
            label, confidence = predict_sign(model, roi)
            if label is None or confidence < args.threshold:
                continue

            text = f'{label} ({confidence:.2f})'
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            last_sign = text
            history.appendleft(f'[{time.strftime("%H:%M:%S")}] {text}')
            confidence_log.append(confidence)
            count += 1

        h, w = frame.shape[:2]
        panel = create_dashboard(args.panel_width, h)
        draw_panel(panel, last_sign, history, count, confidence_log)
        combined = np.hstack([frame, panel])

        curr_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        cv2.putText(combined, f'FPS: {int(fps)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow('Traffic Sign Dashboard', combined)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()