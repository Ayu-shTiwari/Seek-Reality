# utils.py

import tensorflow as tf
import numpy as np
import csv
import os
import pandas as pd
from models.config import FEATURE_STORE_PATH, CLASS_NAMES, AUTO_RETRAIN_PATH, RETRAIN_THRESHOLD, ACCURACY_THRESHOLD
import matplotlib.pyplot as plt

def preprocess_image(image_path, img_size):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path does not exist: {image_path}")
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    return img_array / 255.0

def visualize_probabilities(probabilities, predicted_class=None):
    """
    Visualizes class probabilities as a bar chart.
    Highlights the predicted class in green.
    """
    labels = list(probabilities.keys())
    values = list(probabilities.values())

    # Color the predicted class green, others blue
    colors = ['green' if label == predicted_class else 'skyblue' for label in labels]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=colors)
    plt.xlabel('Class Labels')
    plt.ylabel('Probabilities')
    plt.title('Class Probabilities')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)

    # Annotate each bar with the probability value
    for bar, prob in zip(bars, values):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{prob:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def log_prediction(image_path, predicted_class, confidence):
    """
    Logs the prediction to the feature store CSV and checks if retraining is needed.
    """
    os.makedirs(os.path.dirname(FEATURE_STORE_PATH), exist_ok=True)
    file_exists = os.path.isfile(FEATURE_STORE_PATH)

    # Append to feature store
    with open(FEATURE_STORE_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['image_path', 'predicted_class', 'confidence'])
        writer.writerow([image_path, predicted_class, confidence])

    # After logging, check if auto-retrain conditions are met
    maybe_trigger_auto_retrain()

def maybe_trigger_auto_retrain():
    """
    Check if number of predictions reached threshold and average confidence is low.
    If so, auto_retrain.py is triggered.
    """
    if not os.path.exists(FEATURE_STORE_PATH):
        return

    df = pd.read_csv(FEATURE_STORE_PATH)
    if len(df) < RETRAIN_THRESHOLD:
        return

    avg_confidence = df['confidence'].astype(float).mean()
    if avg_confidence < ACCURACY_THRESHOLD:
        print(f"Triggering auto-retrain: avg confidence={avg_confidence:.2f}")
        os.system(f'python {AUTO_RETRAIN_PATH}')  # Run retrain script
    else:
        print(f"No retrain needed: avg confidence={avg_confidence:.2f}")
