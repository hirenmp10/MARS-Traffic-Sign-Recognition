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

model = tf.keras.models.load_model('traffic_sign_model_clean.keras', compile=False)

def preprocess_differentiable(img_tensor):
    # Differentiable version of the preprocessing.
    # Actually, the model expects an image of shape (1, 32, 32, 3) in range 0-1
    return img_tensor

# Optimize a 32x32 crop to output class 14
img_tensor = tf.Variable(tf.random.uniform((1, 32, 32, 3), 0.0, 1.0))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)

for i in range(100):
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        preds = model(img_tensor, training=False)
        loss = -preds[0, 14] 
        
    grads = tape.gradient(loss, img_tensor)
    optimizer.apply_gradients([(grads, img_tensor)])
    img_tensor.assign(tf.clip_by_value(img_tensor, 0.0, 1.0))
    
    if i % 20 == 0:
        print(f"Iter {i}: Loss = {loss.numpy():.4f}, Conf = {tf.nn.softmax(preds)[0, 14].numpy():.4f}, Raw Pred = {np.argmax(preds.numpy())}")

opt_img = img_tensor.numpy()[0]
opt_img_bgr = (opt_img * 255.0).astype(np.uint8)

# Now we need to undo the histogram equalization somewhat, or just ignore it.
# The `preprocess_image` does:
# 1. Resize to 32x32
# 2. BGR to YUV
# 3. EqualizeHist on Y
# 4. YUV to BGR
# 5. / 255.0

# If we just put this `opt_img_bgr` into the center of a red octagon, it might get mangled by the YUV EQ.
# Let's just create 100 random combinations of large red blobs and save the first one that gives class 14!

from preprocess import preprocess_image, detect_sign_regions
for i in range(1000):
    bg = np.full((300, 300, 3), 200, dtype=np.uint8)
    
    # Draw a random red blob
    radius = np.random.randint(50, 140)
    color = (np.random.randint(0, 50), np.random.randint(0, 50), np.random.randint(150, 255))
    cv2.circle(bg, (150, 150), radius, color, -1)
    
    # Add random text logic
    cv2.putText(bg, "STOP", (100, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), np.random.randint(2, 6))
    
    # Random noise lines
    for _ in range(np.random.randint(0, 10)):
        cv2.line(bg, (np.random.randint(0,300), np.random.randint(0,300)), 
                     (np.random.randint(0,300), np.random.randint(0,300)), 
                     (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)), 2)

    boxes = detect_sign_regions(bg, min_area=300)
    if boxes:
        x, y, w, h, _ = boxes[0]
        roi = bg[y:y+h, x:x+w]
        proc = preprocess_image(roi)
        if proc is not None:
            preds = model.predict(proc, verbose=0)
            cid = int(np.argmax(preds))
            if cid == 14:
                print("FOUND A RED BLOB THAT GIVES STOP!")
                cv2.imwrite("working_stop_blob.jpg", bg)
                break
else:
    print("Failed to find any random blob that gives Stop.")
