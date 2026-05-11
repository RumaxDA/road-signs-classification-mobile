import tensorflow as tf
from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam




BASE_MODEL_PATH = "/home/rumaxx/road-signs-project/ml/new_trained_models/CNN/cnn_gtsrb_best.keras"
DATA_DIR = "/home/rumaxx/road-signs-project/ml/experiments/polish_finetune" 

SELECTED_CLASSES = ['13', '17', '22', '25', '28']
NUM_CLASSES = len(SELECTED_CLASSES)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# === DATA GENERATOR (FILTR KLAS) ===
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    classes=SELECTED_CLASSES,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    classes=SELECTED_CLASSES,
    class_mode='categorical',
    subset='validation'
)

print("Class indices:", train_generator.class_indices)

# === LOAD MODEL ===
base_model = load_model(BASE_MODEL_PATH)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output



x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)

output = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

# === COMPILE ===
model.compile(
    optimizer=Adam(learning_rate=1e-6),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === TRAINING ===
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# === SAVE MODEL ===
SAVE_PATH = "/home/rumaxx/road-signs-project/ml/new_trained_models/cnn_finetuned_selected_classes.keras"

model.save(SAVE_PATH)

print(f"Model zapisany: {SAVE_PATH}")