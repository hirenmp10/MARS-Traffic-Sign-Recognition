# Mobile and Autonomous Robots (UE23CS343BB7)

## Jackfruit Problem (Mini-Project Report)

**Project Title:** Jackfruit Problem - Autonomous Traffic Sign Recognition & Navigation

**Student Name(s) & SRN(s):**

1. Harini Hegde
2. Kaparthy Reddy
3. Hiren MP
4. Ishanvee Amit Sinha

**Instructor:** [Insert Instructor Name]

---

### 1. Introduction
Provide a brief background of the project. 

**The Problem Statement:**
The "Jackfruit Problem" involves the development of a robust vision-and-control pipeline for a mobile robot. The goal is to enable the robot to autonomously navigate a simulated environment (Gazebo) by recognizing traffic signs in real-time and adjusting its linear/angular velocity accordingly.

**Importance in Robotics/Automation:**
Real-time traffic sign recognition (TSR) is a foundational component of Level 4 and Level 5 autonomous vehicles. It ensures compliance with traffic laws and improves safety in dynamic environments where GPS and mapping might be insufficient for local maneuvering.

**Brief Overview of Solution:**
Our solution implements a modular ROS 2 architecture. It consists of a **Vision Node** that uses OpenCV for image preprocessing (HSV masking) and a **Convolutional Neural Network (CNN)** for classification. The classified sign is then published to a **Controller Node**, which maps the sign type to specific `TwistStamped` velocity commands to actuate the robot's movement.

---

### 2. Objective
Clearly state the objectives of the project:

**What are we trying to achieve?**
*   Develop a 4-stage data pipeline: Perception -> Cognition -> Decision -> Actuation.
*   Achieve high accuracy in sign classification using a custom-trained CNN on the GTSRB dataset.
*   Implement a fail-safe heuristic logic to handle visual "hallucinations" or detection noise.

**Expected Outcomes:**
*   Successful identification of Red/Blue traffic signs in Gazebo.
*   Precise robot behavior: Stopping at "Stop" signs and maintaining correct speeds (02, 0.4, 0.6 m/s).
*   A live dashboard for monitoring the robot's visual thought process.

**Scope of the Project:**
The scope is limited to the recognition of standard traffic signs (Stop, Speed Limits, Directions) within a simulated Gazebo environment using ROS 2 Jazzy.

---

### 3. Methodology
Describe the overall approach followed to complete the project.

#### 3.1 Algorithms
**1. Convolutional Neural Network (CNN):**
*   **Name and Description:** A Sequential model with 3x3 kernels for feature extraction, ReLU activation for non-linearity, and MaxPooling for spatial reduction.
*   **Execution Flow:** Image -> Convolution -> ReLU -> MaxPool -> Flatten -> Dense -> Softmax.
*   **Justification:** CNNs are the industry standard for spatial pattern recognition, capable of handling variations in lighting and slight motion blur.

**2. HSV Color Space Masking:**
*   **Description:** Isolates specific color ranges (Red/Blue) to extract Regions of Interest (ROI).
*   **Justification:** HSV is more robust than BGR for color detection in varying lighting conditions, as it isolates Hue (color) from Value (brightness).

**3. Heuristic Error Correction (Rule-Based Validation):**
*   **Description:** A safety layer that overrides the CNN if the classified label contradicts the detected region color.

#### 3.2 Tools Used
*   **Software:** ROS 2 Jazzy, Python 3, OpenCV 4.x, TensorFlow/Keras.
*   **Simulation Environments:** Gazebo Sim (with Simulated Camera Plugin).
*   **Libraries/Frameworks:** NumPy, CvBridge, rclpy.
*   **Hardware:** Simulated Mobile Robot (TurtleBot-style).

---

### 4. Outcome
Explain the final results and achievements of your project.

#### 4.1 Result Screenshots
*Include clear screenshots of:*
1.  **Simulation Result:** [Attach screenshot of robot in Gazebo approaching a sign]
2.  **System Output:** [Attach screenshot of the ROS 2 dashboard showing sign detection]
3.  **Classification:** [Attach image showing the ROI extraction and CNN confidence level]

#### 4.2 Video Demonstration
**Drive Link:** ______________________________________

*Ensure the video clearly demonstrates:*
*   Working of the project (Detection to Movement)
*   Key features (HUD, Side-by-side view)
*   Results (Stop sign behavior)

#### 4.3 GitHub Repository
**GitHub Link:** ______________________________________

*   The repository includes all source code, models, and a detailed README.md.

---

### 5. Conclusion
**Key Learnings:**
*   Mastered ROS 2 modularity and node communication using standard message types.
*   Understood the importance of image preprocessing (HSV masking, Morphological Closing) in vision reliability.
*   Learned to balance model complexity with real-time performance requirements.

**Challenges Faced:**
*   **Lighting variations:** Handled shadows in Gazebo by switching to HSV color space.
*   **Control Latency:** Optimized the ROI processing loop to ensure timely velocity updates.
*   **System Integration:** Managing library dependencies (TensorFlow + ROS 2) in a unified environment.

---
**Submission Instructions:**
*   Submit in PDF format only.
*   File name format: Team-No.pdf.
