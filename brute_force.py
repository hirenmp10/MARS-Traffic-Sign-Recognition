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

img = cv2.imread('stop.jpg')

# Find the bounding box first using red filter
boxes = detect_sign_regions(img, min_area=300)
x, y, w, h, _ = boxes[0]
roi = img[y:y+h, x:x+w]

processed = preprocess_image(roi)

print("Checking permutations:")
perms = [
    [0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]
]
for p in perms:
    perm_img = processed[... , p]
    preds = model.predict(perm_img, verbose=0)
    cid = int(np.argmax(preds))
    print(f"Perm {p} -> {CLASS_LABELS[cid]}")
