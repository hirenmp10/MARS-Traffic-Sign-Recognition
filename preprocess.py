
import cv2
import numpy as np

def preprocess_image(image):
    """
    Standardizes raw BGR image regions for CNN inference.
    
    Processing Steps for Viva:
    1. Resize: Ensures fixed input (32x32) for the Neural Network.
    2. YUV Conversion: Isolates Brightness (Y) from Color (UV).
    3. Histogram Equalization: This is CRITICAL. It normalizes lighting, making the 
       recognition robust against shadows or bright light in the simulation.
    4. Normalization: Scales pixel values [0-255] to [0.0-1.0] to help CNN converge faster.
    """
    if image is None or image.size == 0:
        return None

    # Step 1: Resize to 32x32 (Target size for the CNN model)
    img = cv2.resize(image, (32, 32))

    # Step 2-4: Contrast Enhancement (Luminance Equalization)
    # We convert BGR to YUV to separate light intensity (Y) from color (UV)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    
    # Equalize the histogram of the Y channel to normalize lighting/shadows
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    
    # Convert back to BGR so it's ready for the RGB-trained model
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # Normalize pixel values to 0.0-1.0 (helps neural network weights converge)
    img = img.astype(np.float32) / 255.0
    
    # Add a batch dimension: (32, 32, 3) -> (1, 32, 32, 3)
    return np.expand_dims(img, axis=0)


def detect_sign_regions(frame, min_area=800):
    """
    Extracts candidate traffic sign regions using Color-Based Segmentation.
    
    Algorithmic Logic for Viva:
    1. HSV Space: Unlike BGR, HSV (Hue-Saturation-Value) separates color from light.
       This makes detection stable regardless of brightness.
    2. Dual Masking: 'Red' exists at both ends of the Hue spectrum (0-10 and 170-180).
       We create two masks and combine them using bitwise_or.
    3. Morphology (CLOSE): Uses a kernel to 'fill holes' in the detected blobs.
    4. Contour Detection: Connects adjacent pixels to find the boundaries of the sign.
    5. Bounding Box: Extract (x,y,w,h) for the CNN to focus only on that region.
    """
    h_frame, w_frame = frame.shape[:2]
    # Convert to HSV (Hue, Saturation, Value) 
    # Why? It is robust against lighting changes compared to RGB/BGR
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red color range detection: Red wraps around the 0-180 scale
    # Mask 1: Lower Red (0-10 deg), Mask 2: Upper Red (170-180 deg)
    mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(mask1, mask2)

    # Morphological CLOSE: Dilation followed by Erosion
    # This fills small "holes" inside the detected red objects
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find the external boundaries of the white regions in our mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area: # Filter out tiny noise blobs
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Safety Clipping: Ensure coordinates don't go outside image frame
        x, y = max(0, x), max(0, y)
        w = min(w, w_frame - x)
        h = min(h, h_frame - y)

        if w > 0 and h > 0:
            boxes.append((x, y, w, h, int(area)))

    # ROI Prioritization: Largest sign first (heuristic for 'nearest sign')
    boxes.sort(key=lambda b: b[4], reverse=True)
    return boxes