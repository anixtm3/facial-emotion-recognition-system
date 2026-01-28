# Facial Recognition System (CNN + OpenCV)
## Overview
This project implements a **Facial Emotion Recognition System** using **Deep Learning (CNN)** and **Computer Vision**.  
The system detects a human face from a webcam feed and classifies the facial expression into one of seven emotions in real time.
**Recognized emotions:**
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

---
## Objectives
- To understand and implement the **Convolutional Neural Networks (CNNs)**
- To perform **image-based emotion classification**
- To build a **real-time emotion recognition system**
- To gain hands-on experience with **TensorFlow, Keras and OpenCV**

---

## How the system works
1. Facial images are used to train a CNN model
2. Images are preprocessed (grayscale, normalized)
3. The CNN learns emotion-specific facial features
4. The trained model is saved
5. During runtime:
    - Webcam captures frames
    - Face is detected
    - Emotion is predicted by the trained model
    - Emotion label is displayed in real time

---

## Project Structure
```
facial-emotion-recognition/
│
├── dataset/                     # Training dataset
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
│
├── train_emotion_model.py       # Model training script
├── emotion_webcam.py            # Real-time emotion detection
├── emotion_model.h5             # Trained CNN model
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation

```

---

## Dataset Description
The dataset used for training the model consists of facial images with the following properties:
- Image size: 48 × 48 pixels
- Color format: Grayscale
- One face per image
- Images organized into folders based on emotion labels

### Dataset Source
The dataset was obtained from Kaggle:

https://www.kaggle.com/datasets/msambare/fer2013

The dataset is **not included** in this repository due to size and licensing constraints.

---

## System Architecture
The system follows the pipeline below:

1. Image normalization and preprocessing
2. Feature extraction using Convolutional Neural Networks
3. Emotion classification using a softmax output layer
4. Real-time inference using webcam input

---

## Model Architecture
The CNN model consists of:
- Three convolutional layers with ReLU activation
- Batch Normalization layers for training stability
- Max Pooling layers for dimensionality reduction
- Dropout layers to reduce overfitting
- Fully connected dense layers
- Softmax output layer for multi-class classification

The model outputs probabilities for each of the seven emotion classes.


---

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy

---

## Installation Instructions
### Step 1: Clone the repository
```bash
git clone https://github.com/aniketrepo/facial-recognition-system.git
cd facial-recognition-system
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

---


