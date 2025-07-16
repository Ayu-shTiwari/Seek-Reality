import os
import pandas as pd
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from models.data_handler import get_datasets
from models.model_resnet import build_resnet_model
from models.model_xception import build_xception_model
from models.mesonet import build_meso_model
from models.config import (
    FEATURE_STORE_PATH,
    RETRAIN_AFTER_SAMPLES,
    VAL_PATH,
    IMG_SIZE,
    BATCH_SIZE,
    NUM_CLASSES,
    MODEL_PATHS, 
    CLASS_NAMES,
    PATIENCE,
    EPOCHS
)

# Save state of last retrain
LAST_RETRAIN_FILE = "feature_store/last_retrain_count.txt"

def read_last_retrain_count():
    if os.path.exists(LAST_RETRAIN_FILE):
        with open(LAST_RETRAIN_FILE, 'r') as file:
            return int(file.read().strip())
    return 0

def write_last_retrain_count(count):
    with open(LAST_RETRAIN_FILE, 'w') as file:
        file.write(str(count))

def count_feature_entries():
    if os.path.exists(FEATURE_STORE_PATH):
        df = pd.read_csv(FEATURE_STORE_PATH)
        return len(df)
    return 0

def evaluate_model(model, val_dataset):
    loss, accuracy = model.evaluate(val_dataset, verbose=0)
    return accuracy * 100

def retrain_model(name, builder_func, save_path, train_ds, val_ds):
    model = builder_func(input_shape=IMG_SIZE + (3,), num_classes=NUM_CLASSES)

    callbacks = [
        EarlyStopping(patience=PATIENCE, restore_best_weights=True, verbose=1),
        ModelCheckpoint(save_path, save_best_only=True, verbose=1)
    ]

    print(f"\n[INFO] Retraining {name}...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2)

    val_acc = evaluate_model(model, val_ds)
    print(f"[INFO] New validation accuracy for {name}: {val_acc:.2f}%")
    return val_acc

def main():
    current_count = count_feature_entries()
    last_count = read_last_retrain_count()

    new_entries = current_count - last_count
    print(f"[INFO] New predictions since last retrain: {new_entries}")

    if new_entries >= RETRAIN_AFTER_SAMPLES:
        train_ds, val_ds, _ = get_datasets()

        resnet_val_acc = evaluate_model(load_model(MODEL_PATHS['resnet50']), val_ds)
        xception_val_acc = evaluate_model(load_model(MODEL_PATHS['xception']), val_ds)

        print(f"[INFO] Current ResNet val acc: {resnet_val_acc:.2f}%")
        print(f"[INFO] Current Xception val acc: {xception_val_acc:.2f}%")

        if resnet_val_acc < 90:
            retrain_model("ResNet50", build_resnet_model, MODEL_PATHS['resnet50'], train_ds, val_ds)

        if xception_val_acc < 90:
            retrain_model("Xception", build_xception_model, MODEL_PATHS['xception'], train_ds, val_ds)

        retrain_model("MesoNet", build_meso_model, MODEL_PATHS['mesonet'], train_ds, val_ds)

        write_last_retrain_count(current_count)
    else:
        print("[INFO] Not enough new predictions to retrain yet.")

if __name__ == "__main__":
    main()
    print("[INFO] Auto retrain script finished.")