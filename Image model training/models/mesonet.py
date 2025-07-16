from tensorflow.keras import layers, models, optimizers
from models.config import IMG_SIZE, NUM_CLASSES, LEARNING_RATE

def build_meso_model():
    model = models.Sequential([
        layers.Conv2D(8, (3, 3), activation='relu', input_shape=IMG_SIZE + (3,)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(8, (5, 5), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(16, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model