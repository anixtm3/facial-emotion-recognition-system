# Facial Recognition System
A real-time deep learning system that classifies emotions from webcam feed using CNNs.

## Table of Contents
- [Overview](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#overview)
- [Objectives](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#objectives)
- [How the system works](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#how-the-system-works)
- [Project Structure](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#project-structure)
- [Dataset Description](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#dataset-description)
	- [Dataset Source](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#dataset-source)
- [System Architecture](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#system-architecture)
- [Model Architecture](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#model-architecture)
- [Technologies Used](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#technologies-used)
- [Python Version Requirement](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#python-version-requirement)
- [Installation Instructions](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#installation-instructions)
	- [Step 1](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#step-1-clone-the-repository)
	- [Step 2](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#step-2-install-dependencies)
	- [Step 3](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#step-3-install-dependencies)
	- [Step 4](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#step-4-dataset-setup)
	- [Step 5](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#step-5-train-emotion-recognition-model-required)
	- [Step 6](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#step-6-run-the-system)
	- [Step 7](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#step-6-run-the-system)
- [Output](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#output)
- [Performance Notes](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#performance-notes)
- [Limitations](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#limitations)
- [Future Enhancements](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#future-enhancements)
- [Conclusion](https://github.com/aniketrepo/facial-recognition-system?tab=readme-ov-file#conclusion)

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

## Objectives
- To understand and implement the **Convolutional Neural Networks (CNNs)**
- To perform **image-based emotion classification**
- To build a **real-time emotion recognition system**
- To gain hands-on experience with **TensorFlow, Keras and OpenCV**

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

## Project Structure
```bash
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

## System Architecture
The system follows the pipeline below:
1. Image normalization and preprocessing
2. Feature extraction using Convolutional Neural Networks
3. Emotion classification using a softmax output layer
4. Real-time inference using webcam input

## Model Architecture
The CNN model consists of:
- Three convolutional layers with ReLU activation
- Batch Normalization layers for training stability
- Max Pooling layers for dimensionality reduction
- Dropout layers to reduce overfitting
- Fully connected dense layers
- Softmax output layer for multi-class classification

The model outputs probabilities for each of the seven emotion classes.

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy

## Python Version Requirement
This project is tested and verified on **Python 3.10.11**.

**Important:**  
Some libraries used in this project (such as `face-recognition` and its dependencies like `dlib`) are **not fully compatible with Python 3.11+** at the time of development.

Using a Python version other than **3.10.11** may result in installation errors or runtime issues.

**Recommended version:**  
- Python **3.10.11**

## Installation Instructions
### Step 1: Clone the repository

```bash
git clone https://github.com/aniketrepo/facial-recognition-system.git
cd facial-recognition-system
```

### Step 2: Create a Virtual Environment
Windows
```python
python -m venv venv
venv\Scripts\activate
```

Linux/MacOS
```python
python3.10 -m venv venv
source venv/bin/activate
```
> Make sure the Python version inside venv is 3.10.11

### Step 3: Install Dependencies
```python
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Dataset Setup
- Add face images to the `dataset/` directory
- Each emotion should have **it's own folder**
```shell
emotion_dataset/
├── angry/
├── happy/
├── sad/
├── neutral/
```

### Step 5: Train Emotion Recognition Model (Required)
This step generated the trained model file (`.h5`).
```python
python train_emotion_model.h5
```
**Output:**
```text
emotion_model.h5
```
> This step is required only once unless you change the dataset.

### Step 6: Run the System
```python
python webcam_recognition.py
```

### Step 7: Controls
- Press `Q` to quit the application

## Output
- Detected face is highlighted using a bounding box
- Predicted emotion label is displayed above the face
- Predictions are based on the highest probability from the model output

## Performance Notes
- Accuracy depends on dataset size and class balance
- Similar emotions such as fear and surprise may overlap
- Lighting conditions affect face detection performance
- Emotion recognition is probabilistic and not always exact

## Limitations
- Works best with frontal faces
- Sensitive to lighting and camera quality
- Does not account for head pose variations
- Emotion classification may vary across individuals 

## Future Enhancements
- Data augmentation to improve accuracy
- Confusion matrix and detailed evaluation metrics
- Graphical User Interface
- Integration with face recognition
- Deployment as a web or desktop application

## Conclusion
This project demonstrates a complete deep learning-based solution for facial emotion recognition. It includes dataset preparation, CNN training, model evaluation, and real-time emotion prediction using webcam input.
