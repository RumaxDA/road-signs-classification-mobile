import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

# === KONFIGURACJA GPU ===
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Znaleziono {len(gpus)} GPU – memory growth włączony.")
    except RuntimeError as e:
        print(e)

# === STAŁE ===
NUM_CATEGORIES = 43
TEST_CSV_PATH = '/home/rumaxx/road-signs-project/ml/data/Test.csv'
TEST_ROOT_DIR = '/home/rumaxx/road-signs-project/ml/data'

PLOTS_DIR = '/home/rumaxx/road-signs-project/ml/new_plots/EfficientNet/tl_96'
os.makedirs(PLOTS_DIR, exist_ok=True)
BATCH_SIZE = 32

IMG_HEIGHT, IMG_WIDTH = 96, 96

# === WYBÓR MODELU ===
MODEL_PATH = '/home/rumaxx/road-signs-project/ml/new_trained_models/TL/tl_96/efficientnet_b0_96_v1.keras'
HISTORY_PATH = '/home/rumaxx/road-signs-project/ml/new_trained_models/TL/tl_96/efficientnet_b0_96_history_v1.json'

# === FUNKCJE POMOCNICZE (WYKRESY) ===
def plot_history(history, name):
    acc = history.get('accuracy', history.get('acc', []))
    val_acc = history.get('val_accuracy', history.get('val_acc', []))
    loss = history.get('loss', [])
    val_loss = history.get('val_loss', [])

    if not acc:
        print("Pusta historia lub zły format kluczy.")
        return

    plt.figure(figsize=(12, 5))
    
    # Wykres Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title(f"{name} - Accuracy")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    
    # Wykres Loss
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title(f"{name} - Loss")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, f"{name}_history.png")
    plt.savefig(save_path)
    plt.close()
    print(f"✓ Zapisano wykres historii: {save_path}")

def plot_conf_mat(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 20)) 
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=range(NUM_CATEGORIES), yticklabels=range(NUM_CATEGORIES))
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.title(f"{name} - Confusion Matrix", fontsize=20)
    
    save_path = os.path.join(PLOTS_DIR, f"{name}_confusion_matrix.png")
    plt.savefig(save_path)
    plt.close()
    print(f"✓ Zapisano macierz pomyłek: {save_path}")

# === PREPROCESSING ===
def load_crop_and_preprocess(path, x1, y1, x2, y2, label):
    # 1. Wczytanie
    file_content = tf.io.read_file(path)
    img = tf.io.decode_image(file_content, channels=3, expand_animations=False)
    
    # 2. Bezpieczne koordynaty (Slicing Logic)
    shape = tf.shape(img)
    src_h, src_w = shape[0], shape[1]

    x1 = tf.clip_by_value(tf.cast(x1, tf.int32), 0, src_w - 1)
    y1 = tf.clip_by_value(tf.cast(y1, tf.int32), 0, src_h - 1)
    x2 = tf.clip_by_value(tf.cast(x2, tf.int32), x1 + 1, src_w)
    y2 = tf.clip_by_value(tf.cast(y2, tf.int32), y1 + 1, src_h)

    # 3. Wycięcie (Slicing)
    img_cropped = img[y1:y2, x1:x2, :]

    # 4. Standardowy Resize
    img_resized = tf.image.resize(img_cropped, (IMG_HEIGHT, IMG_WIDTH))
    
    # 5. Konwersja na float32 - BRAK DZIELENIA PRZEZ 255.0 DLA EFFICIENTNET
    img_final = tf.cast(img_resized, tf.float32)

    return img_final, label

# === NORMALIZOWANA CONFUSION MATRIX ===
def plot_conf_mat_normalized(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    plt.figure(figsize=(20,20))
    sns.heatmap(cm, cmap="Blues")

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"{name} - Normalized Confusion Matrix")

    save_path = os.path.join(PLOTS_DIR, f"{name}_confusion_matrix_normalized.png")
    plt.savefig(save_path)
    plt.close()

    print(f"✓ Zapisano znormalizowaną macierz: {save_path}")


#=== WYKRES PEWNOŚCI ===
def plot_confidence_distribution(y_true, y_pred, y_pred_probs, name):
    confidences = np.max(y_pred_probs, axis=1)

    correct_conf = confidences[y_true == y_pred]
    incorrect_conf = confidences[y_true != y_pred]
    
    # zmiana

    avg_correct = np.mean(correct_conf) if len(correct_conf) > 0 else 0
    avg_incorrect = np.mean(incorrect_conf) if len(incorrect_conf) > 0 else 0
    
    print(f"\nStatystyki pewności (Confidence):")
    print(f"Średnia pewność przy poprawnych: {avg_correct:.4f}")
    print(f"Średnia pewność przy błędnych:   {avg_incorrect:.4f}")

    # zmiana

    plt.figure(figsize=(10, 6))


    bins = np.linspace(0.0, 1.0, 40)
    
    plt.hist(correct_conf, bins=bins, alpha=0.6, color='green', label='Poprawne predykcje')
    
    if len(incorrect_conf) > 0:
        plt.hist(incorrect_conf, bins=bins, alpha=0.8, color='red', label='Błędne predykcje')

    plt.xlabel('Pewność modelu (Softmax Probability)', fontsize=12)
    plt.ylabel('Liczba próbek', fontsize=12)
    plt.yscale('log')
    plt.title(f'{name} - Rozkład pewności siebie (Confidence Distribution)', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(axis='y', alpha=0.3)

    save_path = os.path.join(PLOTS_DIR, f"{name}_confidence_distribution.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Zapisano histogram pewności: {save_path}")

    return avg_correct, avg_incorrect

# === NAJCZESCIEJ MYLONE KLASY === 
def analyze_misclassifications(y_true, y_pred, top_n=5):
    cm = confusion_matrix(y_true, y_pred)
    mis = []

    for i in range(NUM_CATEGORIES):
        for j in range(NUM_CATEGORIES):
            if i != j and cm[i,j] > 0:
                mis.append((i,j,cm[i,j]))

    mis_sorted = sorted(mis, key=lambda x: x[2], reverse=True)[:top_n]

    print("\nNajczęściej mylone klasy:")
    for true, pred, count in mis_sorted:
        print(f"True {true} → Pred {pred} : {count}")
    
    return mis_sorted

# === TOP-K ACCURACY ===
def compute_topk(y_true, y_pred_probs):
    top1 = np.mean(np.argmax(y_pred_probs, axis=1) == y_true)
    top3 = np.mean([y_true[i] in np.argsort(y_pred_probs[i])[-3:] for i in range(len(y_true))])
    top5 = np.mean([y_true[i] in np.argsort(y_pred_probs[i])[-5:] for i in range(len(y_true))])
    print("\nTop-k Accuracy:")
    print(f"Top-1: {top1:.4f}")
    print(f"Top-3: {top3:.4f}")
    print(f"Top-5: {top5:.4f}")
    return top1, top3, top5

# === MAIN ===
if __name__ == "__main__":

    filename = os.path.basename(MODEL_PATH)
    print(f"=== ROZPOCZYNAM EWALUACJĘ MODELU: {filename} ===")

    model_display_name = "EfficientNetB0_96"


    # Wczytanie CSV
    print('Wczytywanie ścieżek z Test.csv...')
    try:
        df = pd.read_csv(TEST_CSV_PATH)
        df = df.sort_values(by=['Path']) 
        
        image_paths = [os.path.join(TEST_ROOT_DIR, p) for p in df["Path"]]
        y_true = df["ClassId"].values
        rois = df[['Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2']].values
        print(f"Załadowano {len(y_true)} próbek testowych.")
    except Exception as e:
        print(f"Błąd wczytywania {TEST_CSV_PATH}: {e}"); exit()

    # Tworzenie Datasetu
    print("Tworzenie generatora danych testowych...")
    AUTOTUNE = tf.data.AUTOTUNE
    test_ds = tf.data.Dataset.from_tensor_slices((
        image_paths, 
        rois[:, 0], rois[:, 1], rois[:, 2], rois[:, 3], 
        y_true
    ))
    test_ds = test_ds.map(load_crop_and_preprocess, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    # Wczytywanie modelu
    print(f'Wczytywanie modelu z {MODEL_PATH}...')
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model załadowany pomyślnie.")
    except Exception as e:
        print(f"KRYTYCZNY BŁĄD: Nie można wczytać modelu.\n{e}"); exit()

    # Wczytaj historię i rysuj wykresy
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r") as f:
                history = json.load(f)
                plot_history(history, model_display_name)
        except Exception as e:
            print(f"Nie udało się wczytać historii: {e}")
    else:
        print(f"Brak pliku historii: {HISTORY_PATH}")

    # Ewaluacja
    print("\n=== WYNIKI NA ZBIORZE TESTOWYM ===")
    results = model.evaluate(test_ds, verbose=0)
    print(f"Test Loss:     {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")

    # Predykcje i czas
    print("\nGenerowanie predykcji...")
    t0 = time.perf_counter()
    y_pred_probs = model.predict(test_ds, verbose=0)
    t1 = time.perf_counter()
    
    total_time = t1 - t0
    per_image_ms = (total_time / len(y_true)) * 1000
    print(f"Średni czas inferencji: {per_image_ms:.3f} ms / zdjęcie")

    y_pred = np.argmax(y_pred_probs, axis=1)

    # Raport klasyfikacji
    print("\n=== Szczegółowy Raport Klasyfikacji ===")
    report = classification_report(y_true, y_pred, zero_division=0)
    print(report)

    # Macierz pomyłek
    plot_conf_mat(y_true, y_pred, model_display_name)
    plot_conf_mat_normalized(y_true, y_pred, model_display_name)


    # Analiza najczęstszych błędów i Top-k
    misclassified = analyze_misclassifications(y_true, y_pred)
    top1, top3, top5 = compute_topk(y_true, y_pred_probs)

    # ===== ZAPIS DO PLIKU =====
    avg_corr, avg_incorr = plot_confidence_distribution(y_true, y_pred, y_pred_probs, model_display_name)

    report_path = os.path.join(PLOTS_DIR, f"{model_display_name}_full_report.txt")
    with open(report_path,"w") as f:
        # NOWE METRYKI W RAPORCIE
        f.write("=== Ogólne Wyniki Modelu ===\n")
        f.write(f"Test Loss:           {results[0]:.4f}\n")
        f.write(f"Test Accuracy:       {results[1]:.4f}\n")
        f.write(f"Total Infer. Time:   {total_time:.4f} s\n")
        f.write(f"Time per Image:      {per_image_ms:.3f} ms\n\n")
        
        f.write("=== Classification Report ===\n")
        f.write(report)
        f.write("\n\n=== Top-k Accuracy ===\n")
        f.write(f"Top-1: {top1:.4f}\n")
        f.write(f"Top-3: {top3:.4f}\n")
        f.write(f"Top-5: {top5:.4f}\n")

        f.write("\n=== Analiza Pewności (Confidence) ===\n")
        f.write(f"Avg Confidence (Correct):   {avg_corr:.4f}\n")
        f.write(f"Avg Confidence (Incorrect): {avg_incorr:.4f}\n")

        if avg_incorr > 0.80:
            f.write("UWAGA: Model wykazuje silny OVERCONFIDENCE przy błędach!\n")

        # Najczęściej mylone klasy
        f.write("\n=== Najczęściej mylone klasy ===\n")
        for true, pred, count in misclassified:
            f.write(f"True {true} → Pred {pred} : {count}\n")

    print(f"✓ Raport zapisany: {report_path}")