# dataset_handler.py
import tensorflow as tf
import os
from models.config import DATASET_PATH, IMG_SIZE, BATCH_SIZE, AUGMENTATION

def validate_dataset_structure(dataset_path):
    print(f"Validating dataset structure at {dataset_path}...")
    required_folders = ['Test', 'Train', 'Validation']
    for folder in required_folders:
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Missing required folder: {folder} in {dataset_path}")
        for subfolder in ['Real', 'Fake']:
            if not os.path.exists(os.path.join(folder_path, subfolder)):
                raise FileNotFoundError(f"Missing required subfolder: {subfolder} in {folder_path}")

# Corrected and improved dataset loading

def load_datasets():
    validate_dataset_structure(DATASET_PATH)

    def load_balanced_subset(directory):
        try:
            real_ds = tf.keras.preprocessing.image_dataset_from_directory(
                directory,
                labels='inferred',
                label_mode='int',
                class_names=['Real'],
                image_size=IMG_SIZE,
                batch_size=BATCH_SIZE
            )  # Limit to 10,000 images from 'Real'

            fake_ds = tf.keras.preprocessing.image_dataset_from_directory(
                directory,
                labels='inferred',
                label_mode='int',
                class_names=['Fake'],
                image_size=IMG_SIZE,
                batch_size=BATCH_SIZE
            )  # Limit to 10,000 images from 'Fake'

            return real_ds.concatenate(fake_ds)  
        except Exception as e:
            raise RuntimeError(f"Error loading dataset from {directory}: {e}")

    train_ds = load_balanced_subset(DATASET_PATH + '/Train')
    val_ds = load_balanced_subset(DATASET_PATH + '/Validation')

    try:
        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            DATASET_PATH + '/Test',
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE
        )
    except Exception as e:
        raise RuntimeError(f"Error loading test dataset: {e}")

    if AUGMENTATION:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.1),
        ])
        train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

    print("Datasets loaded successfully.")
    print(f"Train dataset size: {len(train_ds)} batches")
    print(f"Validation dataset size: {len(val_ds)} batches")
    print(f"Test dataset size: {len(test_ds)} batches")

    return train_ds, val_ds, test_ds

if __name__ == "__main__":
    train_ds, val_ds, test_ds = load_datasets()

