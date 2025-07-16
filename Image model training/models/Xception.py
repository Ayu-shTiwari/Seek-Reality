# models/xception_model.py

from tensorflow.keras.applications import Xception
from tensorflow.keras import layers, models, optimizers
from models.config import IMG_SIZE, NUM_CLASSES, LEARNING_RATE

def build_xception_model():
    base_model = Xception(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    base_model.trainable = False  # Freeze base model

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
