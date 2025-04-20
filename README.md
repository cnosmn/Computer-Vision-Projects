# OpenCV Projects Collection

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![YOLOv5](https://img.shields.io/badge/YOLOv5-6.0-red.svg)](https://github.com/ultralytics/yolov5)

## Overview

This repository contains a collection of computer vision applications and projects using OpenCV and related libraries. From basic image processing operations to advanced object detection implementations, this repository serves as both a learning resource and a practical toolset for computer vision tasks.

## Repository Structure

```
opencv-projects/
│
├── opencv_basics/                # OpenCV fundamentals and techniques
│   ├── basic_operation/          # Core image operations
│   ├── blurring/                 # Image smoothing techniques
│   ├── threshold/                # Thresholding methods
│   ├── edge_detection/           # Edge detection algorithms
│   ├── drawing/                  # Drawing shapes and annotations
│   └── contours/                 # Contour detection and analysis
│
├── opencv_random_image/          # Random image processing utilities
│
├── project1_face_detection/      # Face detection implementation
│
├── project2_phoneDetection/      # Phone object detection project
│
└── project3_yolov5/              # YOLOv5 object detection implementation
```

## OpenCV Basics

The `opencv_basics` directory contains fundamental techniques and operations in computer vision:

### Basic Operations
- Image loading, displaying, and saving
- Color space conversions (RGB, BGR, HSV, Grayscale)
- Image cropping and resizing
- Pixel manipulation

### Blurring
- Gaussian blur
- Median blur
- Bilateral filtering
- Box filter

### Thresholding
- Binary thresholding
- Adaptive thresholding
- Otsu's method
- Color-based thresholding

### Edge Detection
- Sobel edge detector
- Canny edge detector
- Laplacian edge detection
- Gradient magnitude and direction

### Drawing
- Drawing shapes (lines, rectangles, circles)
- Adding text to images
- Drawing contours and polygons

### Contours
- Contour detection
- Contour properties (area, perimeter)
- Contour approximation
- Shape analysis

## OpenCV Random Image

The `opencv_random_image` directory contains utilities for generating and processing random images, useful for testing and experimentation.

## Project 1: Face Detection

A complete face detection application that:
- Detects human faces in images and video streams
- Uses Haar Cascades or deep learning-based detectors
- Provides bounding box visualization
- Supports multiple face detection

## Project 2: Phone Detection

An object detection system specifically tuned for detecting mobile phones in images and videos:
- Custom-trained models for phone detection
- Real-time detection capabilities
- Phone orientation and position analysis
- Demonstration of practical object detection use cases

## Project 3: YOLOv5

Implementation of the state-of-the-art YOLOv5 object detection model:
- Pre-trained model usage
- Custom model training pipeline
- Real-time object detection
- Multi-class detection with high accuracy

## Getting Started

### Prerequisites

- Python 3.6 or higher
- OpenCV 4.x
- NumPy
- PyTorch (for YOLOv5)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/cnosmn/Computer-Vision-Projects.git
cd Computer-Vision-Projects
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

### Running the Basic Examples

Navigate to the `opencv_basics` directory and run any of the Python scripts:

```bash
cd opencv_basics/basic_operation
python read_display_image.py
```

### Running the Projects

Each project directory contains its own instructions and requirements. General steps:

1. Navigate to the project directory:
```bash
cd project1_face_detection
```

2. Run the main script:
```bash
python face_detection.py
```

## Project-Specific Information

### Face Detection

The face detection project demonstrates:
- Haar Cascade classifiers
- Face detection in still images
- Real-time face detection from webcam
- Detection parameter tuning

### Phone Detection

This project focuses on:
- Custom object detection for mobile phones
- Detection in various lighting conditions
- Model optimization for real-time performance
- Integration with other systems

### YOLOv5

The YOLOv5 implementation covers:
- Model inference with pre-trained weights
- Custom dataset preparation
- Training on custom object classes
- Deployment considerations for real-time applications

## Usage Examples

### Basic OpenCV Operations

```python
import cv2
import numpy as np

# Read an image
img = cv2.imread('sample.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Display the result
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Face Detection

```python
import cv2

# Load the cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the input image
img = cv2.imread('people.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangles around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the output
cv2.imshow('Detected Faces', img)
cv2.waitKey(0)
```

## Future Development

Planned additions to this repository:
- More advanced image processing techniques
- Machine learning integration with computer vision
- Video analysis and tracking systems
- AR/VR application examples
- Integration with cloud vision APIs

## Contribution

Contributions to this project are welcome! Please feel free to submit a Pull Request.

## Resources and Further Reading

- [OpenCV Documentation](https://docs.opencv.org/)
- [YOLOv5 GitHub Repository](https://github.com/ultralytics/yolov5)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Computer Vision: Algorithms and Applications](http://szeliski.org/Book/)