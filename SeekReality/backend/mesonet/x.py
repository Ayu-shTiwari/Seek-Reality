import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from mesonet.model import make_prediction

def predict_deepfake(model_path, image_dir, thresholds=[0.3, 0.5, 0.7]):
    """
    Predict whether images in a directory are deepfake or real using a pre-trained model.
    Tests multiple thresholds and label inversion.
    """
    for threshold in thresholds:
        print(f"\nTesting with threshold = {threshold}")
        print("=" * 50)
        
        # Test with normal labels
        print("Normal Labels:")
        result, report = make_prediction(
            model_path=model_path,
            data_dir=image_dir,
            threshold=threshold,
            batch_size=1,
            return_probs=True,
            return_report=True
        )
        for row in result:
            filename, label, prob = row
            print(f"Image: {filename}")
            print(f"Predicted Label: {label}")
            print(f"Probability: {float(prob):.4f}%")
            print("-" * 30)
        
        # Test with inverted labels
        print("Inverted Labels:")
        result_inv, _ = make_prediction(
            model_path=model_path,
            data_dir=image_dir,
            threshold=threshold,
            batch_size=1,
            return_probs=True,
            return_report=False,
            invert_labels=True
        )
        for row in result_inv:
            filename, label, prob = row
            print(f"Image: {filename}")
            print(f"Predicted Label: {label}")
            print(f"Probability: {float(prob):.4f}%")
            print("-" * 30)
        
        if report:
            print("Classification Report (Normal Labels):")
            print(report)

def main():
    model_path = 'weights.hdf5'
    image_dir = 'test_images'
    
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        return
    if not os.path.exists(image_dir):
        print(f"Image directory not found at {image_dir}")
        return
    
    try:
        predict_deepfake(model_path, image_dir)
    except Exception as e:
        print(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()