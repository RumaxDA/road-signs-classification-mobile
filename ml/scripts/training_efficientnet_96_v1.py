import os
import json
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.applications import EfficientNetB0
from keras.layers import (Input, Dense, Dropout, GlobalAveragePooling2D, 
                          BatchNormalization,
                          RandomRotation, RandomZoom, RandomTranslation, RandomContrast)
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# === CONFIG ===
TRAIN_CSV_PATH = "/home/rumaxx/road-signs-project/ml/data/Train.csv"
ROOT_DIR = "/home/rumaxx/road-signs-project/ml/data"
SAVE_DIR = "/home/rumaxx/road-signs-project/ml/new_trained_models/experimental"
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_HEIGHT = 32
IMG_WIDTH = 32
NUM_CLASSES = 43
BATCH_SIZE = 32
EPOCHS = 30
SEED = 42
LR = 1e-3

MODEL_PATH = os.path.join(SAVE_DIR, "efficientnet_b0_32_v2.keras")
HISTORY_PATH = os.path.join(SAVE_DIR, "efficientnet_b0_32_history_v2.json")

# === DATA LOADING ===
def load_and_preprocess(path, x1, y1, x2, y2, label):
    file_content = tf.io.read_file(path)
    img = tf.io.decode_image(file_content, channels=3, expand_animations=False)
    
    img = img[y1:y2, x1:x2, :]
    img = tf.image.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    

    img = tf.cast(img, tf.float32) 
    return img, label

def create_dataset(df, training=True):
    paths = [os.path.join(ROOT_DIR, p) for p in df["Path"]]
    labels = df["ClassId"].values
    rois = df[['Roi.X1','Roi.Y1','Roi.X2','Roi.Y2']].values

    ds = tf.data.Dataset.from_tensor_slices(
        (paths, rois[:,0], rois[:,1], rois[:,2], rois[:,3], labels)
    )
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(2000)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

def build_efficientnet():
    base_model = EfficientNetB0(
        weights='imagenet', 
        include_top=False, 
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    
    base_model.trainable = False 

    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    x = base_model(inputs, training=False)

    # 3. Nowa głowa klasyfikująca (Top)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    
    return Model(inputs, outputs)

# === TRAINING ===
if __name__ == "__main__":
    print("=== START TRAINING: Transfer Learning - EfficientNetB0 ===")

    df = pd.read_csv(TRAIN_CSV_PATH)
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["ClassId"], random_state=SEED
    )

    train_ds = create_dataset(train_df, True)
    val_ds = create_dataset(val_df, False)

    model = build_efficientnet()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        ModelCheckpoint(filepath=MODEL_PATH, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1), 
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
    with open(HISTORY_PATH, "w") as f:
        json.dump(history_dict, f)

    print(f"Model zapisany: {MODEL_PATH}")