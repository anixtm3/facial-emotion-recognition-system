import json
import time
import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize


# CONFIGURATION
IMG_SIZE = 48
BATCH_SIZE = 64
DATASET_PATH = "dataset"
MODEL_PATH = "emotion_model.h5"
HISTORY_PATH = "training_history.json"

OUTPUT_DIR = "evaluation"

LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


# CREATE OUTPUT DIRECTORY
os.makedirs(OUTPUT_DIR, exist_ok=True)


# LOAD TRAINING HISTORY
print("\nLoading training history...")

with open(HISTORY_PATH, "r") as f:
    history = json.load(f)

epochs = range(1, len(history["accuracy"]) + 1)


# TRAINING ACCURACY GRAPH
plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    epochs,
    history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(OUTPUT_DIR, "training_accuracy.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: training_accuracy.png")


# TRAINING LOSS GRAPH
plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    history["loss"],
    label="Training Loss"
)

plt.plot(
    epochs,
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(OUTPUT_DIR, "training_loss.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: training_loss.png")


# LOAD MODEL
print("\nLoading trained model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully.")


# CREATE VALIDATION DATASET
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)


# MODEL PREDICTIONS
print("\nGenerating predictions...")

predictions = model.predict(
    val_data,
    verbose=1
)

predicted_classes = np.argmax(predictions, axis=1)
true_classes = val_data.classes

print("Predictions generated.")


# CONFUSION MATRIX
print("\nGenerating confusion matrix...")

cm = confusion_matrix(
    true_classes,
    predicted_classes
)

plt.figure(figsize=(9, 8))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=LABELS
)

disp.plot(
    cmap="Blues",
    values_format="d"
)

plt.title("Confusion Matrix")
plt.xticks(rotation=45)

plt.savefig(
    os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: confusion_matrix.png")


# CLASSIFICATION METRICS
print("\nClassification Report:")
print("=" * 70)

report = classification_report(
    true_classes,
    predicted_classes,
    target_names=LABELS,
    digits=4
)

print(report)


# Extract metrics for graph
report_dict = classification_report(
    true_classes,
    predicted_classes,
    target_names=LABELS,
    output_dict=True
)

precision = [
    report_dict[label]["precision"]
    for label in LABELS
]

recall = [
    report_dict[label]["recall"]
    for label in LABELS
]

f1 = [
    report_dict[label]["f1-score"]
    for label in LABELS
]


# PRECISION / RECALL / F1 BAR CHART
x = np.arange(len(LABELS))
width = 0.25

plt.figure(figsize=(12, 7))

plt.bar(
    x - width,
    precision,
    width,
    label="Precision"
)

plt.bar(
    x,
    recall,
    width,
    label="Recall"
)

plt.bar(
    x + width,
    f1,
    width,
    label="F1-score"
)

plt.xlabel("Emotion")
plt.ylabel("Score")
plt.title("Precision, Recall and F1-score by Emotion")

plt.xticks(
    x,
    LABELS,
    rotation=45
)

plt.ylim(0, 1)
plt.legend()
plt.grid(axis="y")

plt.savefig(
    os.path.join(OUTPUT_DIR, "classification_metrics.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: classification_metrics.png")


# AUC-ROC
print("\nGenerating AUC-ROC graph...")

true_one_hot = label_binarize(
    true_classes,
    classes=np.arange(len(LABELS))
)

plt.figure(figsize=(10, 8))

auc_scores = {}

for i, label in enumerate(LABELS):

    fpr, tpr, _ = roc_curve(
        true_one_hot[:, i],
        predictions[:, i]
    )

    auc_score = auc(fpr, tpr)

    auc_scores[label] = auc_score

    plt.plot(
        fpr,
        tpr,
        label=f"{label} (AUC = {auc_score:.3f})"
    )


# Random classifier reference line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUC-ROC Curves")
plt.legend(loc="lower right")
plt.grid(True)

plt.savefig(
    os.path.join(OUTPUT_DIR, "auc_roc.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: auc_roc.png")


# PRINT AUC SCORES
print("\nAUC Scores:")
print("=" * 40)

for label, score in auc_scores.items():
    print(f"{label:10s}: {score:.4f}")

macro_auc = np.mean(list(auc_scores.values()))

print("-" * 40)
print(f"Macro AUC  : {macro_auc:.4f}")


# INFERENCE EFFICIENCY
print("\nMeasuring inference efficiency...")

# Take one sample image
sample_image = np.expand_dims(
    val_data[0][0][0],
    axis=0
)

# Warm-up predictions
for _ in range(10):
    model.predict(sample_image, verbose=0)


NUM_TESTS = 200

start_time = time.perf_counter()

for _ in range(NUM_TESTS):
    model.predict(
        sample_image,
        verbose=0
    )

end_time = time.perf_counter()

total_time = end_time - start_time

average_latency = (
    total_time / NUM_TESTS
)

fps = 1.0 / average_latency

print(f"Total inference time : {total_time:.4f} seconds")
print(f"Average latency      : {average_latency * 1000:.2f} ms")
print(f"Estimated FPS        : {fps:.2f}")


# EFFICIENCY GRAPH
metrics = [
    "Latency (ms)",
    "FPS"
]

values = [
    average_latency * 1000,
    fps
]

fig, ax1 = plt.subplots(figsize=(9, 6))

ax1.bar(
    ["Latency"],
    [average_latency * 1000],
    label="Latency"
)

ax1.set_ylabel("Latency (ms)")
ax1.set_title("Model Inference Efficiency")

ax2 = ax1.twinx()

ax2.bar(
    ["FPS"],
    [fps],
    alpha=0.7,
    label="FPS"
)

ax2.set_ylabel("Frames / Predictions per Second")

plt.savefig(
    os.path.join(OUTPUT_DIR, "efficiency.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: efficiency.png")


# FINAL SUMMARY
print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print(f"Validation Accuracy : {np.mean(predicted_classes == true_classes):.4f}")
print(f"Macro AUC           : {macro_auc:.4f}")
print(f"Average Latency     : {average_latency * 1000:.2f} ms")
print(f"Estimated FPS       : {fps:.2f}")

print("\nGraphs saved in:")
print(f"  {OUTPUT_DIR}/")