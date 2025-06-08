# Sign Language Recognition System

This project is a real-time Sign Language Recognition system designed to assist communication for individuals with hearing or speech impairments.

## Overview

The system captures hand gestures via a webcam and translates them into readable text and speech using deep learning. It is built to work on a regular PC and deployable on Raspberry Pi.

## Features

- Real-time sign recognition
- Text output of detected gestures
- Optional voice feedback using text-to-speech
- Lightweight deployment with TensorFlow Lite

## Technologies Used

- Python
- OpenCV
- TensorFlow / TensorFlow Lite
- NumPy
- Raspberry Pi (for hardware version)
- Pyttsx3 (for voice)

## Installation

1. Clone the repository:
   git clone https://github.com/grc0801/sign-language-detection.git

2. Navigate to the project folder:
   cd sign-language-detection

3. Install dependencies:
   pip install -r requirements.txt

## Usage

After installation, run the application using your preferred Python environment or interface.

Make sure the webcam is connected and functional. The system will process the input gestures and provide real-time output in text and optionally voice.

## Note

- Use TensorFlow Lite version for better performance on Raspberry Pi.
- Ensure camera permissions are granted if running on restricted systems.

## Acknowledgments

Inspired by the need for inclusive communication technology.
