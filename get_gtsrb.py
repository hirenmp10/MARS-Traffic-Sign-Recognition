import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import cv2
import numpy as np
import tensorflow as tf

original_init = tf.keras.layers.Dense.__init__
def new_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    original_init(self, *args, **kwargs)
tf.keras.layers.Dense.__init__ = new_init

import tensorflow_datasets as tfds

model = tf.keras.models.load_model('traffic_sign_model_clean.keras', compile=False)

def build_bgr_test_image(img_rgb):
    # The detector will do hsv masks to find red.
    # We should put the GTSRB stop sign in the middle of a white image.
    h, w = img_rgb.shape[:2]
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    bg = np.full((400, 400, 3), 200, dtype=np.uint8)
    y_off = (400 - h) // 2
    x_off = (400 - w) // 2
    bg[y_off:y_off+h, x_off:x_off+w] = img_bgr
    return bg

from preprocess import preprocess_image

# Let's download a small subset of GTSRB
ds = tfds.load('gtsrb', split='train', shuffle_files=False)
count = 0
for example in ds:
    if example['label'].numpy() == 14:  # Stop sign
        img = example['image'].numpy()
        
        test_img = build_bgr_test_image(img)
        cv2.imwrite(f"gtsrb_stop_{count}.jpg", test_img)
        
        # Manually test the exact crop
        processed = preprocess_image(img)
        if processed is not None:
            preds = model.predict(processed, verbose=0)
            cid = int(np.argmax(preds))
            print(f"Stop sign {count} raw crop -> Predicted class {cid}")
            if cid == 14:
                print("FOUND A WORKING RAW CROP!")
                
        count += 1
        if count >= 10:
            break
