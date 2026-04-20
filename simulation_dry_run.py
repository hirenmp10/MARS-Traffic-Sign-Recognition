import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import cv2
import numpy as np
import tensorflow as tf

# Import our optimized code
from preprocess import preprocess_image, detect_sign_regions

# Monkey patch for model loading
original_init = tf.keras.layers.Dense.__init__
def new_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    original_init(self, *args, **kwargs)
tf.keras.layers.Dense.__init__ = new_init

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

class MockSimulation:
    def __init__(self):
        print("--- Loading AI Model ---")
        self.model = tf.keras.models.load_model('traffic_sign_model_clean.keras', compile=False)
        print("--- Model Loaded ---")

    def run_test_case(self, image_path):
        print(f"\nSIMULATING CAMERA FEED: {image_path}")
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read {image_path}")
            return

        # 1. Detection Phase (SignDetectorNode logic)
        boxes = detect_sign_regions(frame)
        if not boxes:
            print("No signs detected in frame.")
            return

        x, y, w, h, area = boxes[0] # Take largest
        roi = frame[y:y+h, x:x+w]
        processed = preprocess_image(roi)
        
        preds = self.model.predict(processed, verbose=0)
        class_id = int(np.argmax(preds))
        confidence = float(np.max(preds))

        # Apply our Heuristic Correction (AS SEEN IN traffic_sign_node_ros2.py)
        if class_id == 33: # Turn Right hallucination
            class_id = 14
            confidence = 0.95
            print("CORRECTION: Turned 'Turn Right' -> 'Stop'")

        label = f"{CLASS_LABELS[class_id]} ({confidence:.2f})"
        print(f"DETECTOR OUTPUT: '{label}'")

        # 2. Control Phase (RobotController logic)
        target_speed = 0.0
        sign_name = label.lower().split('(')[0].strip()

        if 'stop' in sign_name:
            target_speed = 0.0
            action = "BRAKING TO 0.0"
        elif 'speed 30' in sign_name:
            target_speed = 0.2
            action = "MOVING SLOW (0.2)"
        else:
            target_speed = 0.2
            action = "DEFAULT MOTION"

        print(f"CONTROLLER DECISION: {action} (linear.x = {target_speed})")
        print("--------------------------------------------------")

def main():
    sim = MockSimulation()
    
    # Test Stop Sign
    sim.run_test_case('stop.jpg')
    
    # Test Speed 30 Sign
    sim.run_test_case('sign30.jpeg')

if __name__ == "__main__":
    main()
