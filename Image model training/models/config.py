# config.py
import os
import importlib

# Dataset Configuration
DATASET_PATH = r'Dataset'  # Should contain Train/Validation/Test subfolders
TRAIN_PATH = os.path.join(DATASET_PATH, 'Train')
VAL_PATH = os.path.join(DATASET_PATH, 'Validation')
TEST_PATH = os.path.join(DATASET_PATH, 'Test')
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
BATCH_SIZE = 32
NUM_CLASSES = 2
CLASS_NAMES = ['Real', 'Fake']

# Training Configuration
EPOCHS = 10
LEARNING_RATE = 0.0001
AUGMENTATION = True
PREFETCH = True
EARLY_STOPPING = True
PATIENCE = 3
SEED = 42
AUTOTUNE = 'auto'
USE_MIXED_PRECISION = False

# Model Save Paths
MODEL_DIR = r'Dataset\model_result'
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATHS = {
    'mesonet': os.path.join(MODEL_DIR, 'mesonet_model.keras'),
    'xception': os.path.join(MODEL_DIR, 'xception_model.keras'),
    'resnet50': os.path.join(MODEL_DIR, 'resnet50_model.keras')
}

# LSTM Model Configuration
LSTM_MODEL_DIR = 'video_detection'
LSTM_EPOCHS = 5
LSTM_MODEL_PATH = os.path.join(LSTM_MODEL_DIR, 'lstm_model.h5')

# Feature Store Configuration
FEATURE_STORE_PATH = 'feature_store/features_log.csv'
os.makedirs(os.path.dirname(FEATURE_STORE_PATH), exist_ok=True)
FEATURE_VECTOR_STORE = 'feature_store/vector_features.csv'  # optional for advanced feature logging

# Logging & Monitoring
LOGGING = True
LOG_DIR = 'logs'
TENSORBOARD_LOGS = os.path.join(LOG_DIR, 'tensorboard')
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOGS, exist_ok=True)

# Continual Learning & Retraining Configuration
ENABLE_CONTINUAL_LEARNING = True
RETRAIN_THRESHOLD = 50                # Trigger retrain after 50 predictions
ACCURACY_THRESHOLD = 0.90            # Trigger retrain if confidence < 90%
AUTO_RETRAIN_PATH = 'auto_retrain.py'  # Script to run for auto retraining

# Class Labels for Prediction Display
CLASS_LABELS = CLASS_NAMES  # Compatibility alias
print("Configuration loaded successfully.")