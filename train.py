import os
import glob
import numpy as np
import tensorflow as tf

# ------------------------------
# Set the dataset directory.
# ------------------------------
dataset_dir = "dataset"

# ------------------------------
# Get gesture categories from folder names.
# ------------------------------
gesture_labels = sorted([d for d in os.listdir(dataset_dir)
                         if os.path.isdir(os.path.join(dataset_dir, d))])
if not gesture_labels:
    raise ValueError("No gesture categories found in dataset directory.")
print("Gesture categories:", gesture_labels)
label_to_index = {label: i for i, label in enumerate(gesture_labels)}

# ------------------------------
# Load samples.
# ------------------------------
data = []
labels = []
for label in gesture_labels:
    pattern = os.path.join(dataset_dir, label, "*.txt")
    for file in glob.glob(pattern):
        with open(file, "r") as f:
            content = f.read().strip().split(",")
            # Expect 126 keypoint values + 1 label = 127 tokens.
            if len(content) != 127:
                continue
            keypoints = content[:-1]  # Exclude the stored label.
            try:
                keypoints = [float(val) for val in keypoints]
            except ValueError:
                continue
            data.append(keypoints)
            labels.append(label_to_index[label])

data = np.array(data)   # Shape: (num_samples, 126)
labels = np.array(labels)
print("Data shape:", data.shape)
print("Labels shape:", labels.shape)
if data.shape[0] == 0:
    raise ValueError("Empty dataset.")

# ------------------------------
# Data Augmentation: add Gaussian noise.
# ------------------------------
def augment_data(data, labels, factor=2, noise_std=0.02):
    new_data = []
    new_labels = []
    for sample, label in zip(data, labels):
        for _ in range(factor):
            noise = np.random.normal(0, noise_std, sample.shape)
            new_data.append(sample + noise)
            new_labels.append(label)
    return np.array(new_data), np.array(new_labels)

augment_factor = 2
aug_data, aug_labels = augment_data(data, labels, factor=augment_factor, noise_std=0.02)
print("Augmented data shape:", aug_data.shape)

# Combine original and augmented data.
data = np.concatenate([data, aug_data], axis=0)
labels = np.concatenate([labels, aug_labels], axis=0)
print("Combined data shape:", data.shape)

# ------------------------------
# Shuffle the dataset.
# ------------------------------
perm = np.random.permutation(data.shape[0])
data = data[perm]
labels = labels[perm]

# ------------------------------
# Normalize data.
# ------------------------------
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
std[std == 0] = 1.0
data_norm = (data - mean) / std

# Save normalization parameters and gesture labels for inference.
np.savez("norm_params.npz", mean=mean, std=std)
with open("gesture_labels.txt", "w") as f:
    for label in gesture_labels:
        f.write(label + "\n")
print("Normalization parameters and gesture labels saved.")

# ------------------------------
# Build the model.
# ------------------------------
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(126,)),  # Input shape (126,) for two-hand features.
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(gesture_labels), activation="softmax")
])
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# ------------------------------
# Train the model with EarlyStopping.
# ------------------------------
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
history = model.fit(data_norm, labels,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stop])

# ------------------------------
# Convert the model to TensorFlow Lite.
# ------------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("gesture_model.tflite", "wb") as f:
    f.write(tflite_model)
print("TFLite model saved as gesture_model.tflite")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Predict labels on the validation set
val_predictions = model.predict(data_norm[int(0.8*len(data)):])
val_labels = labels[int(0.8*len(labels)):]

# Compute confusion matrix
cm = confusion_matrix(val_labels, np.argmax(val_predictions, axis=1), labels=range(len(gesture_labels)))

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gesture_labels)
disp.plot(cmap="Blues", xticks_rotation="vertical")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")  # Save as image
plt.close()

# Plot Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.savefig("accuracy_graph.png")  # Save as image
plt.close()

# Plot Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.savefig("loss_graph.png")  # Save as image
plt.close()