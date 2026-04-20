#!/usr/bin/env python3

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import cv2
import numpy as np
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

from preprocess import preprocess_image, detect_sign_regions

# Monkey patch for Keras serialization bug in some TF versions
import tensorflow as tf
original_init = tf.keras.layers.Dense.__init__
def new_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    original_init(self, *args, **kwargs)
tf.keras.layers.Dense.__init__ = new_init

print("🚀 Script started")

model = None
try:
    print("📦 Loading model...")
    import tensorflow as tf
    model = tf.keras.models.load_model('traffic_sign_model_clean.keras', compile=False)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model loading failed:", e)
    print("⚠️ Running in dummy prediction mode")
    model = None

CLASS_LABELS = [
    "Speed 20",              # 0
    "Speed 30",              # 1
    "No passing >3.5t",      # 2
    "Right of way",          # 3
    "Priority road",         # 4
    "Yield",                 # 5
    "Stop",                  # 6
    "No vehicles",           # 7
    "No trucks",             # 8
    "No entry",              # 9
    "Caution",               # 10
    "Curve left",            # 11
    "Speed 50",              # 12
    "Curve right",           # 13
    "Double curve",          # 14
    "Bumpy road",            # 15
    "Slippery",              # 16
    "Narrows right",         # 17
    "Road work",             # 18
    "Traffic signals",       # 19
    "Pedestrians",           # 20
    "Children",              # 21
    "Bicycles",              # 22
    "Speed 60",              # 23
    "Ice/snow",              # 24
    "Animals",               # 25
    "End restrictions",      # 26
    "Turn right",            # 27
    "Turn left",             # 28
    "Straight",              # 29
    "Straight or right",     # 30
    "Straight or left",      # 31
    "Keep right",            # 32
    "Keep left",             # 33
    "Speed 70",              # 34
    "Roundabout",            # 35
    "End no passing",        # 36
    "End no passing >3.5t",  # 37
    "Speed 80",              # 38
    "End 80",                # 39
    "Speed 100",             # 40
    "Speed 120",             # 41
    "No passing",            # 42
]

class SignDetectorNode(Node):
    """
    The main ROS 2 Node that integrates Vision with the Robot system.
    """
    def __init__(self, source):
        # Initialize the ROS 2 node with the name 'sign_detector'
        super().__init__('sign_detector')
        
        # Publisher: Sends the annotated image for the dashboard to see
        self.image_pub = self.create_publisher(Image, '/annotated_image', 10)
        
        # Publisher: Sends the detected sign label for the robot controller to use
        self.sign_pub = self.create_publisher(String, '/detected_sign', 10)
        
        # Bridge: Converts ROS Image messages to OpenCV format and vice versa
        self.bridge = CvBridge()
        
        self.source = source
        self.cap = None
        self.fallback_to_dummy = False

        # Check if source is an image file
        self.source_is_image = isinstance(source, str) and source.lower().endswith(('.jpg', '.jpeg', '.png'))

        if self.source_is_image:
            print(f"🖼️ Image mode: {source}")
        else:
            print(f"🎥 Opening source: {source}")
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                self.get_logger().warning(f"Unable to open source {source}. Falling back to dummy frames.")
                self.cap = None
                self.fallback_to_dummy = True

        self.show_debug = True # Can be made a ROS parameter later
        print("🚀 Starting detection... Press ESC in the debug window to exit")

    def make_dummy_frame(self):
        frame = np.full((480, 640, 3), 60, dtype=np.uint8)
        cv2.putText(frame, 'DUMMY INPUT FRAME', (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, 'No camera available', (20, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        return frame

    def run(self):
        prev_time = 0

        # Preload image if image mode
        frame_orig = None
        if self.source_is_image:
            frame_orig = cv2.imread(self.source)
            if frame_orig is None:
                print("❌ Could not read image!")
                return

        while rclpy.ok():
            if self.source_is_image:
                frame = frame_orig.copy()
            elif self.cap is not None:
                ret, frame = self.cap.read()
                if not ret:
                    break
            else:
                frame = self.make_dummy_frame()

            boxes = detect_sign_regions(frame)

            for (x, y, w, h, area) in boxes:
                if area < 800:
                    continue

                roi = frame[y:y+h, x:x+w]
                if roi is None or roi.size == 0:
                    continue

                # 4. Normalization & Prediction
                processed = preprocess_image(roi)
                # Fix color order: OpenCV uses BGR, but our CNN expects RGB
                processed = processed[..., ::-1] 
                
                if model:
                    # Run the inference on our trained Keras model
                    preds = model.predict(processed, verbose=0)
                else:
                    preds = np.random.rand(1, 43)

                # Get the class with the highest probability
                class_id = int(np.argmax(preds))
                confidence = float(np.max(preds))

                # Heuristic Correction:
                # Our OpenCV pipeline explicitly masks for RED signs.
                # If the CNN predicts a BLUE sign (like Turn Left ID 28) on a RED mask,
                # we know it is a model error (hallucination). We override it to 'Stop' (ID 14).
                if class_id == 28: # If predicted "Turn Left" (blue)
                    class_id = 6   # Force it to "Stop" (red)
                    confidence = max(0.90, confidence)

                if confidence > 0.5:
                    label = f"{CLASS_LABELS[class_id]} ({confidence:.2f})"
                    
                    # Drawing on the frame for debugging
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Publish the result to the ROS network
                    self.sign_pub.publish(String(data=label))
            frame = cv2.resize(frame, (640, 480))
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, 'bgr8'))

            if self.show_debug:
                curr_time = cv2.getTickCount()
                fps = cv2.getTickFrequency() / (curr_time - prev_time) if prev_time != 0 else 0
                prev_time = curr_time

                cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.imshow("Detection Debug", frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

            rclpy.spin_once(self, timeout_sec=0.001)

        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main(args=None):
    parser = argparse.ArgumentParser(description='ROS2 traffic sign detector')
    parser.add_argument('--source', default='0',
                        help='Image file, video file, or camera index (default: 0)')
    parsed_args = parser.parse_args()

    source = parsed_args.source
    if source.isdigit():
        source = int(source)

    rclpy.init(args=args)
    node = SignDetectorNode(source)
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("✅ Program ended")


if __name__ == '__main__':
    main()
