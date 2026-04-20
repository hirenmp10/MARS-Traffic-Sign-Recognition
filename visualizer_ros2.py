
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from collections import deque
import time

class DashboardNode(Node):
    def __init__(self):
        super().__init__('sign_dashboard')

        self.bridge = CvBridge()
        self.history = deque(maxlen=8)
        self.last_sign = "Scanning..."
        self.last_status = "STABLE"
        self.count = 0
        self.confidence_log = deque(maxlen=50)

        self.create_subscription(Image, '/annotated_image', self.image_cb, 10)
        self.create_subscription(String, '/detected_sign', self.sign_cb, 10)

        cv2.namedWindow('MARS - Traffic Sign Dashboard', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('MARS - Traffic Sign Dashboard', 900, 600)

    def sign_cb(self, msg):
        data = msg.data
        self.last_sign = data
        timestamp = time.strftime('%H:%M:%S')
        self.history.appendleft(f"[{timestamp}] {data}")
        self.count += 1

        try:
            # Extract confidence value
            if '(' in data:
                conf = float(data.split('(')[1].replace(')', ''))
                self.confidence_log.append(conf)
            else:
                self.confidence_log.append(0.0)
        except Exception:
            pass

    def image_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().warning(f'Image conversion failed: {e}')
            return

        h, w = frame.shape[:2]
        # Sleek dark side-panel
        panel_w = 400
        panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
        panel[:] = (28, 28, 28) # Deep Dark Grey background

        # Header
        cv2.rectangle(panel, (0, 0), (panel_w, 60), (45, 45, 45), -1)
        cv2.putText(panel, 'MARS SYSTEMS V1.0', (20, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 200, 255), 2)
        
        # Status Section
        cv2.putText(panel, 'SYSTEM STATUS:', (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
        status_color = (0, 255, 0) if self.count > 0 else (0, 165, 255)
        cv2.putText(panel, 'ACTIVE / OPTIMIZED', (20, 115),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, status_color, 2)

        # Last Detected
        cv2.line(panel, (20, 140), (panel_w-20, 140), (60, 60, 60), 1)
        cv2.putText(panel, 'LATEST DETECTION:', (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
        
        display_sign = self.last_sign.split('(')[0].strip()
        cv2.putText(panel, display_sign.upper(), (20, 205),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
        
        # Confidence Meter
        if self.confidence_log:
            latest_conf = self.confidence_log[-1]
            conf_bar_w = int(latest_conf * 300)
            cv2.rectangle(panel, (20, 225), (320, 235), (40, 40, 40), -1)
            cv2.rectangle(panel, (20, 225), (20 + conf_bar_w, 235), (0, 255, 100), -1)
            cv2.putText(panel, f'{int(latest_conf*100)}%', (330, 235),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Log History
        cv2.line(panel, (20, 265), (panel_w-20, 265), (60, 60, 60), 1)
        cv2.putText(panel, 'DETECTION LOG:', (20, 295),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
        
        for i, entry in enumerate(self.history):
            text_color = (220, 220, 220) if i == 0 else (120, 120, 120)
            cv2.putText(panel, entry, (20, 325 + i*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)

        # Performance Graph
        graph_y_base = h - 50
        cv2.putText(panel, 'CONFIDENCE HISTORY', (20, graph_y_base - 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
        
        if len(self.confidence_log) > 1:
            points = list(self.confidence_log)[-30:]
            for i in range(len(points)-1):
                x1 = 20 + i * 12
                x2 = 20 + (i+1) * 12
                y1 = graph_y_base - int(points[i] * 80)
                y2 = graph_y_base - int(points[i+1] * 80)
                cv2.line(panel, (x1, y1), (x2, y2), (255, 150, 0), 2)

        # Total Counter
        cv2.rectangle(panel, (panel_w - 100, 80), (panel_w - 20, 125), (40, 40, 40), -1)
        cv2.putText(panel, str(self.count), (panel_w - 90, 115),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(panel, 'TOTAL', (panel_w - 82, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

        combined = np.hstack([frame, panel])
        cv2.imshow('MARS - Traffic Sign Dashboard', combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

    def run(self):
        try:
            rclpy.spin(self)
        finally:
            cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = DashboardNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
