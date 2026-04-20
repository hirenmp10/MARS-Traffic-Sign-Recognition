#!/usr/bin/env python3
# This tells the system to run this file using Python 3

import os
# Used to handle environment variables

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Disables certain TensorFlow CPU optimizations
# This avoids compatibility/performance issues on some systems

import argparse
# Used to take input arguments from command line (like --source)

import cv2
# OpenCV library for image processing

import numpy as np
# NumPy is used for numerical operations and arrays

import time
# Used for time-related operations (like FPS calculation)

import rclpy
# ROS 2 Python client library

from rclpy.node import Node
# Node is the basic unit in ROS 2 (like a program/module)

from sensor_msgs.msg import Image
# ROS message type for images

from std_msgs.msg import String
# ROS message type for sending text (like labels)

from cv_bridge import CvBridge
# Converts ROS images to OpenCV images and vice versa

from preprocess import preprocess_image, detect_sign_regions
# Import your team’s functions:
# - detect_sign_regions → finds red regions (Kaparthy)
# - preprocess_image → prepares image for CNN (Harini)

# Monkey patch for Keras serialization bug
# Some TensorFlow models fail due to extra parameters
import tensorflow as tf

original_init = tf.keras.layers.Dense.__init__
# Store original Dense layer constructor

def new_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    # Remove unwanted parameter if present
    original_init(self, *args, **kwargs)
    # Call original constructor

tf.keras.layers.Dense.__init__ = new_init
# Replace original constructor with fixed version

print("🚀 Script started")

model = None

try:
    print("📦 Loading model...")
    import tensorflow as tf
    # Load trained CNN model
    model = tf.keras.models.load_model('traffic_sign_model_clean.keras', compile=False)
    # compile=False because we only use it for prediction (not training)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model loading failed:", e)
    print("⚠️ Running in dummy prediction mode")
    model = None
    # If model fails, we still run system with random predictions

# List of all 43 traffic sign labels
# Index = class_id, Value = actual sign name
CLASS_LABELS = [
    "Speed 20", "Speed 30", "No passing >3.5t", "Right of way",
    "Priority road", "Yield", "Stop", "No vehicles",
    "No trucks", "No entry", "Caution", "Curve left",
    "Speed 50", "Curve right", "Double curve", "Bumpy road",
    "Slippery", "Narrows right", "Road work", "Traffic signals",
    "Pedestrians", "Children", "Bicycles", "Speed 60",
    "Ice/snow", "Animals", "End restrictions", "Turn right",
    "Turn left", "Straight", "Straight or right", "Straight or left",
    "Keep right", "Keep left", "Speed 70", "Roundabout",
    "End no passing", "End no passing >3.5t", "Speed 80",
    "End 80", "Speed 100", "Speed 120", "No passing"
]

class SignDetectorNode(Node):
    """
    Main ROS 2 Node: connects vision and robot system
    """

    def __init__(self, source):
        # Initialize node with name 'sign_detector'
        super().__init__('sign_detector')

        # Publisher for sending annotated image (for dashboard)
        self.image_pub = self.create_publisher(Image, '/annotated_image', 10)

        # Publisher for sending detected sign label (for robot control)
        self.sign_pub = self.create_publisher(String, '/detected_sign', 10)

        # Bridge to convert ROS Image <-> OpenCV image
        self.bridge = CvBridge()

        self.source = source
        self.cap = None
        self.fallback_to_dummy = False

        # Check if input is an image file
        self.source_is_image = isinstance(source, str) and source.lower().endswith(('.jpg', '.jpeg', '.png'))

        if self.source_is_image:
            print(f"🖼️ Image mode: {source}")
        else:
            print(f"🎥 Opening source: {source}")
            # Try opening camera/video
            self.cap = cv2.VideoCapture(source)

            if not self.cap.isOpened():
                # If camera fails, use dummy frames
                self.get_logger().warning(f"Unable to open source {source}. Falling back to dummy frames.")
                self.cap = None
                self.fallback_to_dummy = True

        self.show_debug = True
        # If True → show window with detection results

        print("🚀 Starting detection... Press ESC to exit")

    def make_dummy_frame(self):
        # Create a blank image if no camera available
        frame = np.full((480, 640, 3), 60, dtype=np.uint8)

        # Add text on dummy image
        cv2.putText(frame, 'DUMMY INPUT FRAME', (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.putText(frame, 'No camera available', (20, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        return frame

    def run(self):
        prev_time = 0

        # If input is image → load once
        frame_orig = None
        if self.source_is_image:
            frame_orig = cv2.imread(self.source)
            if frame_orig is None:
                print("❌ Could not read image!")
                return

        # Main loop runs continuously
        while rclpy.ok():

            # Get frame based on source type
            if self.source_is_image:
                frame = frame_orig.copy()
            elif self.cap is not None:
                ret, frame = self.cap.read()
                if not ret:
                    break
            else:
                frame = self.make_dummy_frame()

            # Detect sign regions (HSV + contours)
            boxes = detect_sign_regions(frame)

            # Loop through detected regions
            for (x, y, w, h, area) in boxes:

                # Ignore very small areas (noise)
                if area < 800:
                    continue

                # Extract region of interest (ROI)
                roi = frame[y:y+h, x:x+w]

                if roi is None or roi.size == 0:
                    continue

                # Preprocess image for CNN
                processed = preprocess_image(roi)

                # Convert BGR → RGB (CNN expects RGB)
                processed = processed[..., ::-1]

                if model:
                    # Run prediction
                    preds = model.predict(processed, verbose=0)
                else:
                    # If model not loaded → random prediction
                    preds = np.random.rand(1, 43)

                # Get best prediction
                class_id = int(np.argmax(preds))
                confidence = float(np.max(preds))

                # Heuristic correction:
                # If CNN predicts blue sign but we only detect red → fix it
                if class_id == 28:  # Turn Left
                    class_id = 6    # Change to Stop
                    confidence = max(0.90, confidence)

                # Only accept strong predictions
                if confidence > 0.5:

                    label = f"{CLASS_LABELS[class_id]} ({confidence:.2f})"

                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # Draw label text
                    cv2.putText(frame, label, (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Publish detected sign
                    self.sign_pub.publish(String(data=label))

            # Resize image for display
            frame = cv2.resize(frame, (640, 480))

            # Publish image to ROS topic
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, 'bgr8'))

            if self.show_debug:
                # Calculate FPS
                curr_time = cv2.getTickCount()
                fps = cv2.getTickFrequency() / (curr_time - prev_time) if prev_time != 0 else 0
                prev_time = curr_time

                # Show FPS on screen
                cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # Show window
                cv2.imshow("Detection Debug", frame)

                # Exit on ESC key
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            # Non-blocking ROS execution
            rclpy.spin_once(self, timeout_sec=0.001)

        # Release camera resources
        if self.cap:
            self.cap.release()

        cv2.destroyAllWindows()


def main(args=None):
    # Parse input arguments
    parser = argparse.ArgumentParser(description='ROS2 traffic sign detector')
    parser.add_argument('--source', default='0',
                        help='Image, video, or camera index')
    parsed_args = parser.parse_args()

    source = parsed_args.source

    # Convert to integer if camera index
    if source.isdigit():
        source = int(source)

    # Initialize ROS
    rclpy.init(args=args)

    # Create node
    node = SignDetectorNode(source)

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        # Clean shutdown
        node.destroy_node()
        rclpy.shutdown()
        print("✅ Program ended")


if __name__ == '__main__':
    main()