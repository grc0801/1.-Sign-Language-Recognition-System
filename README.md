# Sign Language Recognition System

This project is a real-time Sign Language Recognition system designed to assist communication for individuals with hearing or speech impairments.

## Overview

The system captures hand gestures through a webcam or IP camera feed, processes them using MediaPipe for landmark detection, and translates them into readable text and speech using a deep learning model. It is designed to run on regular PCs as well as lightweight platforms like Raspberry Pi using TensorFlow Lite.

## Features

- Real-time sign recognition
- Uses IP camera or built-in webcam as input source
- Gesture tracking with MediaPipe
- Text output of detected signs
- Optional voice feedback using text-to-speech
- Lightweight and portable using TensorFlow Lite

## Technologies Used

- Python  
- MediaPipe (for real-time hand landmark detection)  
- OpenCV  
- TensorFlow / TensorFlow Lite  
- NumPy  
- IP Camera Integration  
- Raspberry Pi (for hardware version)  
- Pyttsx3 (for voice feedback)

## Installation

1. Download or clone the project folder to your local system.

2. Navigate to the project directory.

3. Install required dependencies:
   pip install -r requirements.txt

## Usage

After installing the dependencies, launch the application using your preferred Python environment.

Make sure your IP camera is accessible or your system's webcam is connected. The application will start processing the video stream, detect hand gestures using MediaPipe, and provide real-time text and voice output.

## Results

- Achieved **100% accuracy** during controlled testing on custom dataset.
- Successfully integrated with both **USB webcam** and **IP camera** for gesture capture.
- Delivers low-latency detection suitable for real-time communication needs.

## Note

- For optimal performance on Raspberry Pi, use TensorFlow Lite version.
- Ensure network access to IP camera and camera permissions on the device are enabled.

## Acknowledgments

Inspired by the need for inclusive communication technology.
