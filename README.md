# MARS Traffic Sign Recognition (Jackfruit Problem)

Autonomous vision-and-control pipeline for a mobile robot implemented in ROS 2. The system enables a robot to navigate a simulated environment by recognizing traffic signs in real-time.

## 🚀 Overview
The "Jackfruit Problem" focuses on a 4-stage data pipeline:
1. **Perception**: Real-time image acquisition from the robot's camera.
2. **Cognition**: Preprocessing (HSV masking) and ROI extraction.
3. **Decision**: Deep learning classification using a Convolutional Neural Network (CNN).
4. **Actuation**: Mapping detected signs to `TwistStamped` velocity commands.

## 🛠️ Key Features
- **CNN-based Classification**: Trained on the GTSRB dataset for high accuracy.
- **ROS 2 Modular Architecture**: Clean separation between vision, control, and visualization nodes.
- **Fail-safe Logic**: Rule-based validation to cross-verify CNN predictions with color heuristics.
- **Live Dashboard**: Standalone dashboard (`run_traffic_sign_dashboard.py`) for real-time visual monitoring.

## 📁 Repository Structure
- `traffic_sign_node_ros2.py`: The main vision and decision node.
- `robot_controller.py`: Translates decisions into robot movement.
- `visualizer_ros2.py`: ROS 2 HUD for the camera feed.
- `preprocess.py`: Image processing utilities.
- `traffic_sign_model_clean.keras`: The pre-trained CNN model.
- `run_traffic_sign_dashboard.py`: Non-ROS standalone monitoring tool.

## 🔧 Installation & Setup

### Prerequisites
- ROS 2 (Jazzy/Humble)
- Python 3.10+
- Gazebo Sim

### Python Dependencies
```bash
pip install -r requirements.txt
```

## 📈 Performance
The model is evaluated using the GTSRB test set.

## 👥 Contributors
- Harini Hegde
- Kaparthy Reddy
- Hiren M P
- Ishanvee Amit Sinha
