# Sign Language Recognition System

This project is a real-time Sign Language Recognition system designed to assist communication for individuals with hearing or speech impairments.

## Overview

The system captures hand gestures through a webcam or IP camera feed, detects key landmarks using MediaPipe, and translates them into readable text and speech using a deep learning model. It supports output via standard display, audio feedback, and OLED screen when deployed on Raspberry Pi. The system is designed to work on both laptops and embedded platforms like Raspberry Pi using TensorFlow Lite.

## Features

- Real-time sign recognition
- Uses IP camera or USB webcam as input source
- Gesture tracking with MediaPipe
- Text output of detected signs
- Optional voice feedback using text-to-speech
- OLED display output support (on Raspberry Pi)
- Lightweight deployment using TensorFlow Lite

## Technologies Used

- Python  
- MediaPipe (for real-time hand landmark detection)  
- OpenCV  
- TensorFlow / TensorFlow Lite  
- NumPy  
- IP Camera Integration  
- Raspberry Pi  
- OLED Display (I2C interface)  
- Pyttsx3 (for voice feedback)

## Installation

1. Download or clone the project folder to your local system.

2. Navigate to the project directory.

3. Install required dependencies:
   pip install -r requirements.txt

## Usage

After installing the dependencies, launch the application using your preferred Python environment.

Make sure your IP camera is accessible or your system's webcam is connected. The application will process the live video feed, detect hand gestures using MediaPipe, and provide real-time text and voice output. When used on Raspberry Pi, the recognized sign will also be displayed on an OLED screen.

## Results

- Achieved **100% accuracy** during controlled testing on a custom sign language dataset.
- Verified to work with both **laptop webcams** and **IP camera streams**.
- Successfully tested on **Raspberry Pi 5**, where recognized text was displayed on an **OLED screen** and converted to speech.
- Responsive real-time output suitable for communication applications.

## Note

- For optimal performance on Raspberry Pi, use the TensorFlow Lite version of the model.
- Ensure I2C is enabled for OLED use on Raspberry Pi, and the correct IP stream URL is provided if using an IP camera.
- 
