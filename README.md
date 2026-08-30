# Facial Emotion Recognition System

A real-time deep learning system that detects human faces from a webcam feed and classifies facial expressions into seven emotion categories using a Convolutional Neural Network (CNN).

The system combines **Computer Vision**, **Deep Learning**, and **Model Evaluation** to provide both real-time emotion recognition and quantitative performance analysis.

<details>
  <summary><strong>Table of Contents</strong></summary>

- [Facial Emotion Recognition System](#facial-emotion-recognition-system)
- [Overview](#overview)
- [Objectives](#objectives)
- [Recognized Emotions](#recognized-emotions)
- [How the System Works](#how-the-system-works)
  - [Real-Time Recognition Pipeline](#real-time-recognition-pipeline)
- [Project Structure](#project-structure)
- [Dataset Description](#dataset-description)
  - [Dataset Source](#dataset-source)
- [System Architecture](#system-architecture)
  - [1. Data Processing](#1-data-processing)
  - [2. CNN-Based Feature Extraction and Classification](#2-cnn-based-feature-extraction-and-classification)
  - [3. Real-Time Inference](#3-real-time-inference)
- [Model Architecture](#model-architecture)
    - [Main Components](#main-components)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
  - [Training Accuracy](#training-accuracy)
  - [Training Loss](#training-loss)
  - [AUC-ROC](#auc-roc)
  - [Confusion Matrix](#confusion-matrix)
  - [Classification Metrics](#classification-metrics)
    - [Precision](#precision)
    - [Recall](#recall)
    - [F1-score](#f1-score)
  - [Inference Efficiency](#inference-efficiency)
- [Technologies Used](#technologies-used)
- [Python Version Requirement](#python-version-requirement)
- [Installation Instructions](#installation-instructions)
  - [Step 1: Clone the Repository](#step-1-clone-the-repository)
  - [Step 2: Create a Virtual Environment](#step-2-create-a-virtual-environment)
    - [Windows](#windows)
    - [Linux/macOS](#linuxmacos)
  - [Step 3: Install Dependencies](#step-3-install-dependencies)
  - [Step 4: Dataset Setup](#step-4-dataset-setup)
  - [Step 5: Train the Emotion Recognition Model](#step-5-train-the-emotion-recognition-model)
  - [Step 6: Evaluate the Model](#step-6-evaluate-the-model)
  - [Step 7: Run the Real-Time System](#step-7-run-the-real-time-system)
  - [Step 8: Controls](#step-8-controls)
- [Output](#output)
- [Performance Notes](#performance-notes)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Conclusion](#conclusion)
- [Author](#author)

</details>

# Overview

This project implements a **Facial Emotion Recognition System (FERS)** using **Deep Learning (CNN)** and **Computer Vision**.

The system detects a human face from a webcam feed and classifies the facial expression into one of seven emotions in real time.

The project consists of three major components:

1. **CNN Model Training** – trains a convolutional neural network using labeled facial images.
2. **Real-Time Emotion Recognition** – detects faces from webcam frames and predicts their emotions.
3. **Model Evaluation** – evaluates the trained model using accuracy, AUC-ROC, confusion matrix, precision, recall, F1-score, and inference efficiency.

# Objectives

- To understand and implement **Convolutional Neural Networks (CNNs)**
- To perform **image-based facial emotion classification**
- To build a **real-time emotion recognition system**
- To understand image preprocessing for deep learning
- To gain hands-on experience with **TensorFlow, Keras, OpenCV, NumPy, and Scikit-learn**
- To evaluate the performance of a trained CNN using multiple classification metrics
- To analyze the efficiency of the model during inference

# Recognized Emotions

The system classifies facial expressions into the following seven categories:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

# How the System Works

The overall workflow is:

```text
                    Facial Emotion Dataset
                              │
                              ▼
                       Image Preprocessing
                              │
                    48 × 48 Grayscale Images
                              │
                              ▼
                         CNN Training
                              │
                              ▼
                       Trained CNN Model
                       emotion_model.h5
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
        Real-Time Recognition         Model Evaluation
                │                           │
                ▼                           ├── Accuracy
          Webcam Input                      ├── AUC-ROC
                │                           ├── Confusion Matrix
                ▼                           ├── Precision
          Face Detection                    ├── Recall
                │                           ├── F1-score
                ▼                           └── Efficiency
        Image Preprocessing
                │
                ▼
          CNN Prediction
                │
                ▼
        Emotion Classification
                │
                ▼
       Emotion Displayed on Screen
```

## Real-Time Recognition Pipeline

During real-time operation:

1. The webcam captures a video frame.
2. The frame is converted from BGR to grayscale.
3. OpenCV's Haar Cascade classifier detects faces.
4. Each detected face is cropped from the frame.
5. The face is resized to **48 × 48 pixels**.
6. Pixel values are normalized to the range **0–1**.
7. The processed face is passed to the trained CNN.
8. The CNN produces probability scores for all seven emotions.
9. The emotion with the highest probability is selected.
10. A bounding box and predicted emotion are displayed on the webcam feed.

# Project Structure

```text
facial-emotion-recognition/
│
├── dataset/                     # Facial emotion dataset
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
│
├── evaluation/                  # Generated model evaluation graphs
│   ├── training_accuracy.png
│   ├── training_loss.png
│   ├── auc_roc.png
│   ├── confusion_matrix.png
│   ├── classification_metrics.png
│   └── efficiency.png
│
├── train_emotion_model.py       # CNN training script
├── evaluate_emotion_model.py    # Model evaluation script
├── emotion_webcam.py            # Real-time emotion recognition
├── emotion_model.h5             # Trained CNN model
├── training_history.json        # Training and validation history
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

# Dataset Description

The dataset used for training the model consists of facial images organized into folders according to their corresponding emotion labels.

The images used by the CNN have the following properties:

- Image size: **48 × 48 pixels**
- Color format: **Grayscale**
- Classification type: **7-class emotion classification**
- Images organized into separate directories for each emotion
- Dataset split:
  - **80% training**
  - **20% validation**

The images are normalized using:

```python
rescale=1./255
```

This converts pixel values from the range `0–255` to approximately `0–1`.

## Dataset Source

The dataset was obtained from Kaggle:

https://www.kaggle.com/datasets/msambare/fer2013

The dataset is **not included in this repository** due to its size and licensing considerations.

# System Architecture

The system consists of three major stages.

## 1. Data Processing

Input images are:

- Resized to 48 × 48 pixels
- Converted to grayscale
- Normalized to the range 0–1

## 2. CNN-Based Feature Extraction and Classification

The processed images are passed through multiple convolutional layers.

The CNN automatically learns visual patterns and features associated with the different facial expressions.

## 3. Real-Time Inference

The trained CNN is loaded into the webcam application.

OpenCV detects faces, preprocesses them, and passes them to the CNN for emotion prediction.

# Model Architecture

The CNN consists of three convolutional blocks followed by fully connected layers.

```text
Input
48 × 48 × 1
     │
     ▼
Conv2D
32 filters, 3 × 3
     │
Batch Normalization
     │
Max Pooling
     │
Dropout (0.25)
     │
     ▼
Conv2D
64 filters, 3 × 3
     │
Batch Normalization
     │
Max Pooling
     │
Dropout (0.25)
     │
     ▼
Conv2D
128 filters, 3 × 3
     │
Batch Normalization
     │
Max Pooling
     │
Dropout (0.25)
     │
     ▼
Flatten
     │
     ▼
Dense
256 neurons
     │
Dropout (0.5)
     │
     ▼
Dense
7 neurons
     │
     ▼
Softmax
     │
     ▼
7 Emotion Probabilities
```

### Main Components

- **Conv2D** – extracts spatial features from facial images
- **ReLU activation** – introduces non-linearity
- **Batch Normalization** – improves training stability
- **Max Pooling** – reduces spatial dimensions
- **Dropout** – helps reduce overfitting
- **Flatten** – converts extracted feature maps into a vector
- **Dense layer** – learns higher-level feature relationships
- **Softmax layer** – produces probabilities for the seven emotion classes

The model uses the **Adam optimizer** and **categorical cross-entropy loss**.

# Model Training

The CNN is trained for **15 epochs** using a batch size of **64**.

Training and validation accuracy/loss are stored in:

```text
training_history.json
```

This history is later used to generate training performance graphs.

The trained model is saved as:

```text
emotion_model.h5
```

# Model Evaluation

The trained model is evaluated using the validation dataset.

The evaluation stage generates several graphs that provide a more complete understanding of model performance.

## Training Accuracy

The training accuracy graph compares:

- Training accuracy
- Validation accuracy

This shows how the model's classification performance changes over the training epochs.

## Training Loss

The training loss graph compares:

- Training loss
- Validation loss

This helps identify learning behavior and potential overfitting.

## AUC-ROC

The **AUC-ROC** graph measures the model's ability to distinguish between each emotion class and the remaining classes.

A separate ROC curve is generated for each emotion:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

The **Area Under the Curve (AUC)** is also calculated for each class.

## Confusion Matrix

The confusion matrix shows the relationship between:

- Actual emotion
- Predicted emotion

It helps identify which emotions are correctly classified and which emotions are commonly confused with one another.

## Classification Metrics

The system calculates the following metrics for each emotion:

- **Precision**
- **Recall**
- **F1-score**

A bar chart is generated to compare these metrics across all seven emotion classes.

### Precision

Measures how many predictions for a particular emotion were actually correct.

### Recall

Measures how many actual instances of an emotion were correctly detected.

### F1-score

Provides a combined measure of precision and recall.

## Inference Efficiency

The system also measures model inference performance.

The evaluation includes:

- Average inference latency
- Estimated predictions per second (FPS)

This is particularly relevant because the final system is intended for **real-time webcam operation**.

# Technologies Used

- **Python**
- **TensorFlow**
- **Keras**
- **OpenCV**
- **NumPy**
- **Scikit-learn**
- **Matplotlib**

# Python Version Requirement

This project is tested and verified on:

**Python 3.10.11**

Recommended environment:

```text
Python 3.10.11
```

Using the recommended Python version helps ensure compatibility with the project's TensorFlow, Keras, NumPy, and OpenCV dependencies.

# Installation Instructions

## Step 1: Clone the Repository

```bash
git clone https://github.com/aniketrepo/facial-recognition-system.git
cd facial-recognition-system
```

## Step 2: Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python3.10 -m venv venv
source venv/bin/activate
```

Make sure the Python version inside the virtual environment is **3.10.11**.

## Step 3: Install Dependencies

Upgrade pip:

```bash
pip install --upgrade pip
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Step 4: Dataset Setup

Place the dataset inside the `dataset/` directory.

The directory structure should look like:

```text
dataset/
├── angry/
├── disgust/
├── fear/
├── happy/
├── neutral/
├── sad/
└── surprise/
```

Each folder should contain images belonging to the corresponding emotion.

## Step 5: Train the Emotion Recognition Model

Run:

```bash
python train_emotion_model.py
```

This will:

1. Load the dataset
2. Split the data into training and validation sets
3. Preprocess the images
4. Train the CNN
5. Save the trained model
6. Save the training history

The generated files are:

```text
emotion_model.h5
training_history.json
```

The model needs to be retrained if the dataset or model architecture is changed.

## Step 6: Evaluate the Model

After training, run:

```bash
python evaluate_emotion_model.py
```

This generates the following evaluation graphs:

```text
evaluation/
├── training_accuracy.png
├── training_loss.png
├── auc_roc.png
├── confusion_matrix.png
├── classification_metrics.png
└── efficiency.png
```

The script also prints the evaluation results in the terminal.

## Step 7: Run the Real-Time System

Start the webcam-based emotion recognition system:

```bash
python emotion_webcam.py
```

The application will open the webcam and begin detecting faces and predicting emotions in real time.

## Step 8: Controls

Press:

```text
Q
```

to exit the application.

# Output

During real-time operation:

- A face is detected using OpenCV
- The detected face is highlighted with a bounding box
- The predicted emotion is displayed above the face
- The prediction is based on the highest probability produced by the CNN

Example:

```text
┌───────────────────────┐
│                       │
│       FACE            │
│                       │
│                       │
└───────────────────────┘
        Happy
```

# Performance Notes

The current model was trained for 15 epochs.

The recorded training results show:

- Initial training accuracy: approximately **39.35%**
- Final training accuracy: approximately **66.23%**
- Final validation accuracy: approximately **52.24%**

The difference between training and validation performance indicates that the model shows signs of **overfitting**.

Performance can vary depending on:

- Dataset size
- Dataset balance
- Image quality
- Lighting conditions
- Facial orientation
- Camera quality
- Individual facial characteristics

Emotion recognition is probabilistic and should not be interpreted as a definitive measurement of a person's actual emotional state.

# Limitations

- Works best with relatively frontal faces
- Sensitive to lighting conditions
- Sensitive to camera quality
- Does not explicitly compensate for large head-pose variations
- Similar facial expressions may be difficult to distinguish
- Model performance depends heavily on the quality and distribution of the training dataset
- The current model shows a noticeable gap between training and validation accuracy
- Facial expression classification does not necessarily represent a person's actual emotional state

# Future Enhancements

Potential improvements include:

- Data augmentation
- Hyperparameter optimization
- Improved CNN architecture
- Transfer learning
- Class balancing
- Early stopping
- Learning-rate scheduling
- Improved face detection
- Real-time confidence scores
- Graphical User Interface
- Integration with face recognition
- Web or desktop deployment
- GPU acceleration
- Improved model generalization
- Additional emotion datasets
- Real-time performance monitoring

# Conclusion

This project demonstrates a complete deep learning-based **Facial Emotion Recognition System**.

It covers the complete machine learning pipeline:

```text
Dataset
   ↓
Image Preprocessing
   ↓
CNN Training
   ↓
Trained Model
   ↓
Model Evaluation
   ↓
Real-Time Webcam Recognition
```

The system can recognize seven facial expressions in real time and provides quantitative evaluation through **training curves, AUC-ROC analysis, confusion matrix, precision, recall, F1-score, and inference efficiency measurements**.

The current results also demonstrate the importance of model evaluation, as the difference between training and validation performance highlights areas where the model can be improved.

# Author

**Aniket Dixit**

B.Tech Data Science