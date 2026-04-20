import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import cv2
import numpy as np
import tensorflow as tf

from preprocess import preprocess_image, detect_sign_regions
import json
import h5py

# Monkey patch keras to ignore quantization_config
original_init = tf.keras.layers.Dense.__init__
def new_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    original_init(self, *args, **kwargs)
tf.keras.layers.Dense.__init__ = new_init

def get_model():
    return tf.keras.models.load_model('traffic_sign_model_clean.keras', compile=False)

try:
    model = get_model()
    print("Model loaded.")
except Exception as e:
    print(f"Still failed: {e}")

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

def test_image(image_path, convert_rgb=False):
    print(f"\n--- Testing {image_path} (RGB_convert={convert_rgb}) ---")
    frame = cv2.imread(image_path)
    if frame is None:
        print("Could not read image.")
        return
        
    boxes = detect_sign_regions(frame)
    if not boxes:
        print("No red regions detected.")
        return
        
    for i, (x, y, w, h, area) in enumerate(boxes):
        if area < 800:
            continue
        roi = frame[y:y+h, x:x+w]
        
        if convert_rgb:
            # Maybe the model expects RGB? Let's simulate by flipping BGR to RGB
            # BEFORE preprocess_image, but wait, preprocess_image does BGR->YUV->BGR and returns BGR.
            # So if we want to give model RGB, we should flip after preprocess_image.
            pass
            
        processed = preprocess_image(roi)
        if processed is None:
            continue
        
        if convert_rgb:
            # flip channels of the normalized image
            processed = processed[..., ::-1]
            
        preds = model.predict(processed, verbose=0)
        class_id = int(np.argmax(preds))
        confidence = float(np.max(preds))
        print(f"Box {i} (area {area}): {CLASS_LABELS[class_id]} ({confidence:.2f})")

test_image('stop.jpg')
test_image('sign30.jpeg')

test_image('stop.jpg', convert_rgb=True)
test_image('sign30.jpeg', convert_rgb=True)
