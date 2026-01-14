import cv2
import mediapipe as mp
import numpy as np

def apply_filter(image, filter_id):
    if filter_id == 0:  # Normal
        return image
    elif filter_id == 1:  # Black & White
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif filter_id == 2:  # Cartoon
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(image, 9, 250, 250)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cartoon
    elif filter_id == 3:  # Sepia
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        return cv2.transform(image, kernel)
    elif filter_id == 4:  # Thermal
        return cv2.applyColorMap(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
    elif filter_id == 5:  # Sketch
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255-blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    elif filter_id == 6:  # Blur
        return cv2.GaussianBlur(image, (25, 25), 0)
    elif filter_id == 7:  # Emboss
        kernel = np.array([[ -2, -1, 0], [ -1,  1, 1], [  0,  1, 2]])
        embossed = cv2.filter2D(image, -1, kernel) + 128
        return np.clip(embossed, 0, 255).astype(np.uint8)
    elif filter_id == 8:  # Edge Detection
        edges = cv2.Canny(image, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    elif filter_id == 9:  # Negative (Invert)
        return cv2.bitwise_not(image)
    elif filter_id == 10:  # Color Enhancement
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = cv2.add(hsv[:,:,1], 50)
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return enhanced
    elif filter_id == 11:  # Pixelation
        h, w = image.shape[:2]
        temp = cv2.resize(image, (w//20, h//20), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    elif filter_id == 12:  # Mosaic
        kernel = np.ones((20,20),np.float32)/400
        return cv2.filter2D(image,-1,kernel)
    return image

FILTER_NAMES = [
    "Normal", "B&W", "Cartoon", "Sepia", "Thermal", "Sketch", "Blur", "Emboss",
    "Edges", "Negative", "Color+", "Pixelate", "Mosaic"
]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=2)

def is_pinch(lm_list, threshold=30):
    x1, y1 = lm_list[4][1], lm_list[4][2]
    x2, y2 = lm_list[8][1], lm_list[8][2]
    return np.hypot(x2-x1, y2-y1) < threshold

def get_landmarks_coords(lms, w, h):
    return [[id, int(lm.x * w), int(lm.y * h)] for id, lm in enumerate(lms.landmark)]

cap = cv2.VideoCapture(0)
filter_id = 0
pinch_last = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    pinch = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            lm_list = get_landmarks_coords(hand_landmarks, w, h)
            if is_pinch(lm_list):
                pinch = True
            xs = [pt[1] for pt in lm_list]
            ys = [pt[2] for pt in lm_list]
            cv2.rectangle(frame, (min(xs), min(ys)), (max(xs), max(ys)), (255, 255, 0), 2)
    if pinch and not pinch_last:
        filter_id = (filter_id + 1) % len(FILTER_NAMES)
    pinch_last = pinch

    filtered = apply_filter(frame, filter_id)
    if len(filtered.shape) == 2:
        filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    cv2.putText(filtered, f'Filter: {FILTER_NAMES[filter_id]}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.imshow("Hand Gesture Filter Switch", filtered)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
