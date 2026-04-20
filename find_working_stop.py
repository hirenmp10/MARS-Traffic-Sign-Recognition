import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import cv2
import numpy as np
import tensorflow as tf

from preprocess import preprocess_image, detect_sign_regions

original_init = tf.keras.layers.Dense.__init__
def new_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    original_init(self, *args, **kwargs)
tf.keras.layers.Dense.__init__ = new_init

model = tf.keras.models.load_model('traffic_sign_model_clean.keras', compile=False)

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

def check(filename):
    frame = cv2.imread(filename)
    boxes = detect_sign_regions(frame, min_area=300)
    for i, (x, y, w, h, area) in enumerate(boxes):
        roi = frame[y:y+h, x:x+w]
        
        # Test original
        processed = preprocess_image(roi)
        preds = model.predict(processed, verbose=0)
        c0 = int(np.argmax(preds))
        
        # Test RGB flip
        processed_rgb = processed[..., ::-1]
        preds_rgb = model.predict(processed_rgb, verbose=0)
        c1 = int(np.argmax(preds_rgb))
        
        print(f"File {filename}: BGR_fed -> {CLASS_LABELS[c0]}, RGB_fed -> {CLASS_LABELS[c1]}")

check("synthetic_stop.jpg")
check("stop.jpg")
