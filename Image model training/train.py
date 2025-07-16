# train.py

from data_handler import load_datasets
#from models.resnet import build_resnet_model
#from models.Xception import build_xception_model
from models.mesonet import build_meso_model
from models.config import EPOCHS, MODEL_PATHS
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def train_model(model_builder, model_path):
    train_ds, val_ds, _ = load_datasets()
    model = model_builder()
    callbacks = [
        ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='val_loss', verbose=1),
        EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    ]
    img_history=model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
    print(f"Training history: {img_history.history}")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    print("Training Mesonet model...")
train_model(build_meso_model, MODEL_PATHS['mesonet'])

    #print("Training ResNet model...")
    #train_model(build_resnet_model, MODEL_PATHS['resnet'])
    #print("Training Xception model...")
    #train_model(build_xception_model, MODEL_PATHS['xception'])
print("Training completed.")