import os
import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
from collections import deque
import time
import pyttsx3
from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306
from luma.core.render import canvas

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load normalization parameters
if os.path.exists("norm_params.npz"):
    params = np.load("norm_params.npz")
    mean = params["mean"].astype(np.float32)
    std = params["std"].astype(np.float32)
    print("Normalization parameters loaded.")
else:
    print("Normalization parameters not found.")
    mean, std = None, None

# Load gesture labels
if os.path.exists("gesture_labels.txt"):
    with open("gesture_labels.txt", "r") as f:
        gesture_labels = [line.strip() for line in f if line.strip()]
    print("Gesture labels loaded:", gesture_labels)
else:
    gesture_labels = ["hello", "ok", "peace"]
    print("Using fallback gesture labels:", gesture_labels)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load the TFLite model
model_path = "gesture_model.tflite"
if not os.path.exists(model_path):
    raise FileNotFoundError("TFLite model file not found.")
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("TFLite model loaded, input shape:", input_details[0]['shape'])

# Constants and settings
DEBUG = True
THRESHOLD = 0.8
SMOOTHING_WINDOW = 5

# Initialize text-to-speech engine
engine = pyttsx3.init()
last_spoken_label = None
last_spoken_time = 0

# Initialize sound module
pygame.mixer.init()

# Function to play sound (if required)
def play_sound(signs_to_audio):
    pygame.mixer.music.load(signs_to_audio)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue

# Preprocessing function
def preprocess_keypoints(hand_list):
    combined = []
    if not hand_list:
        combined = [0.0] * 126
    else:
        if len(hand_list) >= 2:
            sorted_hands = sorted(hand_list, key=lambda h: np.mean([lm.x for lm in h.landmark]))
            for h in sorted_hands[:2]:
                kp = []
                for lm in h.landmark:
                    kp.extend([lm.x, lm.y, lm.z])
                combined.extend(kp)
        else:
            h = hand_list[0]
            kp = []
            for lm in h.landmark:
                kp.extend([lm.x, lm.y, lm.z])
            combined.extend(kp)
            combined.extend([0.0] * 63)

    if len(combined) != 126:
        combined += [0.0] * (126 - len(combined))

    arr = np.array(combined, dtype=np.float32)
    if mean is not None and std is not None:
        arr = (arr - mean) / std
    return arr.reshape(1, -1)

# Gesture classification
def classify_gesture(hand_list):
    inp = preprocess_keypoints(hand_list)
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    if DEBUG:
        print("Raw output:", output)
    idx = int(np.argmax(output))
    conf = output[idx]
    if conf < THRESHOLD:
        return "Unknown", conf, output
    return gesture_labels[idx], conf, output

# Initialize input source
pred_queue = deque(maxlen=SMOOTHING_WINDOW)
ip_cam_url = 'http://192.168.207.73:8080/video'
cap = cv2.VideoCapture(ip_cam_url) if ip_cam_url else cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open camera.")

# Initialize the OLED display
serial = i2c(port=1, address=0x3C)  # Adjust address if necessary
oled = ssd1306(serial)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    hand_list = []
    if results.multi_hand_landmarks:
        for h in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)
            hand_list.append(h)

    if hand_list:
        gesture, conf, raw_vals = classify_gesture(hand_list)
        pred_queue.append((gesture, conf))
        gestures = [g for g, c in pred_queue]
        smoothed = max(set(gestures), key=gestures.count)
        display_text = f"{smoothed} ({conf * 100:.1f}%)"

        cv2.putText(frame, display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        current_time = time.time()
        if conf >= THRESHOLD:
            if smoothed != last_spoken_label:
                print("Playing sound for:", smoothed)
                print(smoothed)
                engine.say(smoothed)
                engine.runAndWait()
                last_spoken_label = smoothed
                last_spoken_time = current_time
            elif current_time - last_spoken_time >= 3:
                print(smoothed)
                engine.say(smoothed)
                engine.runAndWait()
                last_spoken_time = current_time

            with canvas(oled) as draw:
                draw.text((0, 0), smoothed, fill="white")
        else:
            pred_queue.clear()
    else:
        pred_queue.clear()

    cv2.imshow("Real-Time Gesture Recognition", frame)
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
