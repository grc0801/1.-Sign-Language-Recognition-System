# 1.-Sign-Language-Recognition-System

This project is a real-time Sign Language Detection system designed to assist communication for individuals with hearing or speech impairments.

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

## Project Structure

sign-language-detection/
├── model/                 # Trained model files  
├── images/                # Sample input gesture images  
├── sign_detector.py       # Main script  
├── utils/                 # Helper functions  
├── README.md

## Installation

1. Clone the repository:
   git clone https://github.com/yourusername/sign-language-detection.git

2. Navigate to the folder:
   cd sign-language-detection

3. Install dependencies:
   pip install -r requirements.txt

## Run

To start the detection system, run:
   python sign_detector.py

## Note

- Ensure your webcam is connected.
- If running on Raspberry Pi, use the TFLite version for better performance.

## Acknowledgments

Inspired by the need for inclusive communication technology.
