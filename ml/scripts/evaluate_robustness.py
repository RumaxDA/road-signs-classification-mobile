import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# === KONFIGURACJA ===
NUM_CATEGORIES = 43
TEST_CSV_PATH = '/home/rumaxx/road-signs-project/ml/data/Test.csv'
TEST_ROOT_DIR = '/home/rumaxx/road-signs-project/ml/data'

MODEL_PATH = '/home/rumaxx/road-signs-project/ml/new_trained_models/TL/tl_224/efficientnet_b0_224_v1.keras'
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32

# KATALOG NA WYNIKI
SAVE_DIR = '/home/rumaxx/road-signs-project/ml/new_plots/robustness'
os.makedirs(SAVE_DIR, exist_ok=True)

# === FUNKCJE DEGRADACJI (OPENCV / NUMPY) ===
def apply_degradation(img_np, deg_type, severity):
    """
    img_np: obraz w formacie float32 [0.0, 1.0]
    severity: 0 (Lekki), 1 (Średni), 2 (Mocny)
    """
    if deg_type == 'noise':
        # Symulacja szumu ISO matrycy (Gaussian Noise)

        # ================================ CONFIG CNN ===================================
        #std_devs = [0.05, 0.15, 0.25]

        # ================================ CONFIG TL ===================================
        std_devs = [12.75, 38.25, 63.75]
        noise = np.random.normal(0, std_devs[severity], img_np.shape)
        img_degraded = img_np + noise
        
    elif deg_type == 'blur':
        # Symulacja rozmycia w ruchu (Motion/Gaussian Blur)
        #kernels = [3, 5, 9] 
        kernels = [15, 23, 43] 
        k = kernels[severity]
        img_degraded = cv2.GaussianBlur(img_np, (k, k), 0)
        
    elif deg_type == 'darkness':
        # Symulacja spadku jasności (Zmierzch/Noc)
        factors = [0.6, 0.35, 0.15]
        img_degraded = img_np * factors[severity]
        
    else:
        img_degraded = img_np

    # zakres [0, 1] CNN
    #return np.clip(img_degraded, 0.0, 1.0).astype(np.float32)

    # zakres [0, 255] TL
    return np.clip(img_degraded, 0.0, 255.0).astype(np.float32)

# === GENEROWANIE OBRAZKA DEBUG ===
def generate_debug_grid(image_path, x1, y1, x2, y2):
    print("Generowanie obrazka debug_grid.jpg...")
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # koordynaty
    h, w = img.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    img_crop = img[y1:y2, x1:x2]
    img_resized = cv2.resize(img_crop, (IMG_WIDTH, IMG_HEIGHT))
    # =========================== Preprocessing CNN ===============================
    #img_base = img_resized.astype(np.float32) / 255.0

    # =========================== Preprocessin TL =================================
    img_base = img_resized.astype(np.float32)

    degradations = ['noise', 'blur', 'darkness']
    labels = ['Szum (Noise)', 'Rozmycie (Blur)', 'Ciemnosc (Darkness)']
    severities = ['Lekki', 'Sredni', 'Mocny']

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    fig.suptitle("Robustness Test - Dowod Wizualny Degradacji", fontsize=16)

    for i, deg_type in enumerate(degradations):
        # Oryginał w pierwszej kolumnie dla CNN
        #axes[i, 0].imshow(img_base)
        
        # Oryginał w pierwszej kolumnie dla TL
        axes[i, 0].imshow(img_base.astype(np.uint8))
        axes[i, 0].set_title("Oryginal" if i == 0 else "")
        axes[i, 0].axis('off')

        # 3 stopnie zepsucia
        for sev in range(3):
            img_deg = apply_degradation(img_base.copy(), deg_type, sev)
            #axes[i, sev+1].imshow(img_deg)
            # zepsucie dla TL
            axes[i, sev+1].imshow(img_deg.astype(np.uint8))
            axes[i, sev+1].set_title(f"{labels[i]}\n{severities[sev]}")
            axes[i, sev+1].axis('off')

    plt.tight_layout()
    debug_path = os.path.join(SAVE_DIR, "robustness_debug.jpg")
    plt.savefig(debug_path, dpi=150)
    plt.close()
    print(f"✓ Zapisano wizualizacje degradacji w: {debug_path}")

# === PREPROCESSING DO TENSORFLOW ===
def load_crop_and_preprocess(path, x1, y1, x2, y2, label):
    file_content = tf.io.read_file(path)
    img = tf.io.decode_image(file_content, channels=3, expand_animations=False)
    
    shape = tf.shape(img)
    src_h, src_w = shape[0], shape[1]

    x1 = tf.clip_by_value(tf.cast(x1, tf.int32), 0, src_w - 1)
    y1 = tf.clip_by_value(tf.cast(y1, tf.int32), 0, src_h - 1)
    x2 = tf.clip_by_value(tf.cast(x2, tf.int32), x1 + 1, src_w)
    y2 = tf.clip_by_value(tf.cast(y2, tf.int32), y1 + 1, src_h)

    img_cropped = img[y1:y2, x1:x2, :]
    img_resized = tf.image.resize(img_cropped, (IMG_HEIGHT, IMG_WIDTH))
    # config cnn
    #img_final = tf.cast(img_resized, tf.float32) / 255.0
    # config tl
    img_final = tf.cast(img_resized, tf.float32)

    return img_final, label

def get_degraded_dataset(df, deg_type, severity):
    paths = [os.path.join(TEST_ROOT_DIR, p) for p in df["Path"]]
    labels = df["ClassId"].values
    rois = df[['Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2']].values

    ds = tf.data.Dataset.from_tensor_slices((paths, rois[:, 0], rois[:, 1], rois[:, 2], rois[:, 3], labels))
    ds = ds.map(load_crop_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # Zastosowanie degradacji używając tf.py_function
    def degrade_wrapper(img, label):
        def _apply(img_np):
            return apply_degradation(img_np, deg_type, severity)
        
        img_degraded = tf.numpy_function(func=_apply, inp=[img], Tout=tf.float32)
        img_degraded.set_shape((IMG_HEIGHT, IMG_WIDTH, 3))
        return img_degraded, label

    ds = ds.map(degrade_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# === MAIN PIPELINE ===
if __name__ == "__main__":
    print(f"=== ROZPOCZYNAM ROBUSTNESS TEST ===")
    
    df = pd.read_csv(TEST_CSV_PATH)
    
    # 1. Generowanie Debug Image 
    sample_row = df.iloc[10]
    sample_path = os.path.join(TEST_ROOT_DIR, sample_row['Path'])
    generate_debug_grid(sample_path, sample_row['Roi.X1'], sample_row['Roi.Y1'], sample_row['Roi.X2'], sample_row['Roi.Y2'])

    # 2. Ładowanie modelu
    print(f"\nWczytywanie modelu z {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Najpierw mierzymy baseline (czyste zdjęcia)
    print("Testowanie Baseline (Oryginalne zdjęcia)...")
    baseline_ds = get_degraded_dataset(df, 'none', 0) 
    _, baseline_acc = model.evaluate(baseline_ds, verbose=0)
    print(f"Baseline Accuracy: {baseline_acc:.4f}")

    degradations = ['noise', 'blur', 'darkness']
    severity_names = ['Lekki', 'Średni', 'Mocny']
    results = {deg: [baseline_acc] for deg in degradations}

    # 3. Pętla Testowa
    for deg in degradations:
        print(f"\n--- Testowanie degradacji: {deg.upper()} ---")
        for sev in range(3):
            print(f"Poziom: {severity_names[sev]}...", end=" ", flush=True)
            ds = get_degraded_dataset(df, deg, sev)
            _, acc = model.evaluate(ds, verbose=0)
            results[deg].append(acc)
            print(f"Accuracy = {acc:.4f}")

    # 4. Rysowanie wykresu
    plt.figure(figsize=(10, 6))
    x_labels = ['Oryginał', 'Lekki', 'Średni', 'Mocny']
    x_ticks = [0, 1, 2, 3]

    colors = {'noise': 'red', 'blur': 'blue', 'darkness': 'gray'}
    labels_pl = {'noise': 'Szum Matrycy', 'blur': 'Rozmycie', 'darkness': 'Spadek jasności (Noc)'}

    for deg in degradations:
        plt.plot(x_ticks, results[deg], marker='o', linewidth=2, color=colors[deg], label=labels_pl[deg])

    plt.xticks(x_ticks, x_labels)
    plt.ylim(0.0, 1.05)
    plt.xlabel('Poziom degradacji obrazu', fontsize=12)
    plt.ylabel('Dokładność Modelu (Accuracy)', fontsize=12)
    plt.title(f'Odporność Modelu ({IMG_WIDTH}x{IMG_HEIGHT}) na Warunki Drogowe', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower left')

    chart_path = os.path.join(SAVE_DIR, "robustness_chart.png")
    plt.savefig(chart_path)
    plt.close()

    print(f"✓ Wykres zapisany w: {chart_path}")

    report_path = os.path.join(SAVE_DIR, "robustness_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== RAPORT: TEST ODPORNOŚCI (ROBUSTNESS TEST) ===\n")
        f.write(f"Model: {os.path.basename(MODEL_PATH)}\n")
        f.write(f"Rozdzielczość: {IMG_WIDTH}x{IMG_HEIGHT}\n")
        f.write(f"Baseline Accuracy (Czyste dane): {baseline_acc:.4f}\n")
        f.write("--------------------------------------------------\n\n")

        for deg in degradations:
            f.write(f"--- ZAKŁÓCENIE: {deg.upper()} ---\n")
            for sev in range(3):
                # results[deg][0] to baseline, indeksy 1,2,3 to kolejne stopnie zepsucia
                f.write(f"Poziom {severity_names[sev]}: {results[deg][sev+1]:.4f}\n")
            f.write("\n")

    print(f"✓ Raport tekstowy zapisany w: {report_path}")

    print(f"\n=== TEST ZAKOŃCZONY ===")