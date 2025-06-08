import os
import cv2
import mediapipe as mp
import numpy as np


def main():
    # Initialize MediaPipe Hands with support for two hands.
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=2,
                           min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    # Ask user for gesture categories.
    num_categories = int(input("Enter the number of gesture categories to collect: "))
    gesture_labels = []
    for i in range(num_categories):
        label = input(f"Enter name for category {i + 1}: ").strip()
        if not label:
            print("Label cannot be empty.")
            return
        gesture_labels.append(label.lower())  # Lowercase for consistency

    # Create label-to-index mapping and folders.
    label_to_index = {label: i for i, label in enumerate(gesture_labels)}
    os.makedirs("dataset", exist_ok=True)
    for label in gesture_labels:
        os.makedirs(os.path.join("dataset", label), exist_ok=True)

    num_samples = int(input("Enter number of samples per gesture (recommended >=100): "))

    print("Starting data collection. Press 'q' to quit at any time.")

    for label in gesture_labels:
        print(f"\nCollecting samples for gesture '{label}'.")
        input(f"Press Enter when ready for '{label}'.")
        sample_count = 0
        while sample_count < num_samples:
            ret, frame = cap.read()
            if not ret:
                print("Could not read frame.")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            combined_keypoints = []
            # We want to collect features from two hands (2 x 63 = 126 features)
            if results.multi_hand_landmarks:
                # Sort detected hands by x-coordinate (for consistency)
                hands_sorted = sorted(results.multi_hand_landmarks, key=lambda h: np.mean([lm.x for lm in h.landmark]))
                # For each of 2 hands
                for i in range(2):
                    if i < len(hands_sorted):
                        landmarks = hands_sorted[i]
                        kp = []
                        for lm in landmarks.landmark:
                            kp.extend([lm.x, lm.y, lm.z])
                        combined_keypoints.extend(kp)
                        # Draw landmarks for visualization.
                        mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                    else:
                        # If less than 2 hands, fill with zeros.
                        combined_keypoints.extend([0.0] * 63)
            else:
                # If no hands detected, just show the frame.
                cv2.imshow("Data Collection", frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                continue

            # Ensure the feature vector is 126 elements long.
            if len(combined_keypoints) != 126:
                combined_keypoints += [0.0] * (126 - len(combined_keypoints))

            # Create a CSV-like string of all keypoints.
            keypoints_str = ",".join([f"{x:.6f}" for x in combined_keypoints])
            # Save file: Features (126 values) + the label index (total tokens = 127).
            filename = os.path.join("dataset", label, f"{label}_{sample_count}.txt")
            with open(filename, "w") as f:
                f.write(keypoints_str + f",{label_to_index[label]}")
            sample_count += 1

            cv2.putText(frame, f"Collecting {label}: {sample_count}/{num_samples}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Data Collection", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                print("Data collection terminated early.")
                cap.release()
                cv2.destroyAllWindows()
                return

        print(f"Collected {sample_count} samples for gesture '{label}'.")
    print("Data collection complete.")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
