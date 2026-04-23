import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

original_init = tf.keras.layers.Dense.__init__
def new_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    original_init(self, *args, **kwargs)
tf.keras.layers.Dense.__init__ = new_init

model = tf.keras.models.load_model('traffic_sign_model_clean.keras', compile=False)

def tf_eval(samples=50):
    ds = tfds.load('gtsrb', split='test', shuffle_files=True)
    count = 0
    correct = 0
    stop_count = 0
    stop_correct = 0

    for example in ds.take(100):
        img = example['image'].numpy()
        label = example['label'].numpy()
        
        # Test direct model prediction using their normalization
        img_resized = cv2.resize(img, (32, 32))
        img_yuv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        processed = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        processed = processed.astype(np.float32) / 255.0
        processed = np.expand_dims(processed, axis=0)
        
        preds = model.predict(processed, verbose=0)
        p_label = int(np.argmax(preds))
        
        if p_label == label:
            correct += 1
            
        if label == 14:
            stop_count += 1
            if p_label == 14:
                stop_correct += 1
            else:
                print(f"GTSRB Stop (14) predicted as {p_label}")
                
        count += 1
        
    print(f"Overall Accuracy on 100 samples: {correct}/{count}")
    print(f"Stop Sign Accuracy: {stop_correct}/{stop_count}")

tf_eval()
