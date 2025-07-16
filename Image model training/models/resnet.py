# models/resnet_model.py

from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models, optimizers
from models.config import IMG_SIZE, NUM_CLASSES, LEARNING_RATE

def build_resnet_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    base_model.trainable = False  

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
