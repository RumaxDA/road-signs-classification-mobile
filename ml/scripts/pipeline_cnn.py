import os
import matplotlib.pyplot as plt

import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

# === KONFIGURACJA ===
IMAGE_PATH = '/home/rumaxx/road-signs-project/ml/images/27.jpg' 
OUTPUT_FILENAME = 'pipeline_result.jpg'

# 1. MODEL YOLO
#YOLO_MODEL_NAME = 'yolov8n.pt'
#YOLO_MODEL_NAME = '/home/rumaxx/road-signs-project/ml/YOLO/YOLOv2-GTSDB/yolo_universal_sign_det/weights/best.pt'
YOLO_MODEL_NAME = '/home/rumaxx/road-signs-project/ml/YOLO/yolo_universal_sign_det/weights/best.pt'

# 2. MODEL CNN
#CNN_MODEL_PATH = '/home/rumaxx/road-signs-project/ml/trained_models/best_model_finetuned.keras'
CNN_MODEL_PATH = '/home/rumaxx/road-signs-project/ml/new_trained_models/CNN/cnn_48/cnn_48_v1.keras'

MIN_CONFIDENCE = 0.8
IMG_SIZE = 48

# SŁOWNIK KLAS 
CLASSES = {
    0: 'Ograniczenie prędkości (20km/h)', 1: 'Ograniczenie prędkości (30km/h)', 
    2: 'Ograniczenie prędkości (50km/h)', 3: 'Ograniczenie prędkości (60km/h)', 
    4: "Ograniczenie prędkości (70km/h)", 5: "Ograniczenie prędkości (80km/h)", 
    6: "Koniec ograniczenia prędkości (80km/h)", 7: "Ograniczenie prędkości (100km/h)", 
    8: "Ograniczenie prędkości (120km/h)", 9: "Zakaz wyprzedzania", 
    10: "Zakaz wyprzedzania przez pojazdy ciężarowe", 11: "Skrzyżowanie z drogą podporządkowaną", 
    12: "Droga z pierwszeństwem", 13: "Ustąp pierwszeństwa", 14: "Stop", 
    15: "Zakaz ruchu", 16: "Zakaz wjazdu pojazdów ciężarowych", 17: "Zakaz wjazdu", 
    18: "Inne niebezpieczeństwo", 19: "Niebezpieczny zakręt w lewo", 
    20: "Niebezpieczny zakręt w prawo", 21: "Podwójny zakręt, pierwszy w lewo", 
    22: "Nierówna droga", 23: "Śliska jezdnia", 24: "Zagrożenie zwężeniem jezdni - prawostronne", 
    25: "Roboty drogowe", 26: "Sygnalizacja świetlna", 27: "Piesi", 
    28: "Dzieci", 29: "Rowerzyści", 30: "Oszronienie jezdni", 
    31: "Dzikie zwierzęta", 32: "Koniec zakazów", 33: "Nakaz jazdy w prawo", 
    34: "Nakaz jazdy w lewo", 35: "Nakaz jazdy prosto", 36: "Nakaz jazdy prosto lub w prawo", 
    37: "Nakaz jazdy prosto lub w lewo", 38: "Nakaz jazdy z prawej strony znaku", 
    39: "Nakaz jazdy z lewej strony znaku", 40: "Rondo", 
    41: "Koniec zakazu wyprzedzania", 42: "Koniec zakazu wyprzedzania przez pojazdy ciężarowe",
}

def preprocess_for_cnn(img):
    img_resized = cv2.resize(img, (48, 48)) 
    img_float = img_resized.astype("float32") / 255.0
    return np.expand_dims(img_float, axis=0)

def run_pipeline():
    print("=== START SYSTEMU DETEKCJI ===")
    
    print(f"1. Ładowanie YOLO...")
    detector = YOLO(YOLO_MODEL_NAME)

    print(f"2. Ładowanie CNN...")
    try:
        classifier = tf.keras.models.load_model(CNN_MODEL_PATH)
    except Exception as e:
        print(f"   -> BŁĄD CNN: {e}"); return

    if not os.path.exists(IMAGE_PATH):
        print(f"Błąd: Nie znaleziono pliku {IMAGE_PATH}"); return
    
    original_img = cv2.imread(IMAGE_PATH)
    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    # Detekcja (Niski próg, łapiemy wszystko co podejrzane)
    results = detector(original_img, conf=0.10, verbose=False)
    detections = results[0].boxes

    if len(detections) == 0:
        print("   -> YOLO nic nie wykryło.")
        return

    # Przygotowanie plotu
    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    ax = plt.gca()

    print(f"Znaleziono {len(detections)} obiektów. Filtrowanie...")

    for i, box in enumerate(detections):
        # yolo_class_id = int(box.cls[0])
        
        # ACCEPTED_IDS = [9, 11] # Traffic Light, Stop Sign
        # if yolo_class_id not in ACCEPTED_IDS:
        #     continue

        # Wymiary
        coords = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = coords
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        box_w = x2 - x1
        box_h = y2 - y1
        if box_h == 0: continue
        
        aspect_ratio = box_w / box_h
        
 
        if aspect_ratio < 0.75 or aspect_ratio > 1.3:
            print(f"   [ODRZUT] Złe proporcje: {aspect_ratio:.2f} (To nie znak)")
            rect_bad = plt.Rectangle((x1, y1), box_w, box_h, fill=False, color='red', linewidth=1)
            ax.add_patch(rect_bad)
            continue

        # Wycięcie
        cropped_sign = img_rgb[y1:y2, x1:x2]
        if cropped_sign.size == 0: continue
        cropped_sign = cv2.GaussianBlur(cropped_sign, (3, 3), 0)
        # CNN
        tensor_crop = tf.convert_to_tensor(cropped_sign, dtype=tf.float32)
        tensor_resized = tf.image.resize(tensor_crop, (IMG_SIZE, IMG_SIZE), method='bicubic')
        input_batch = tensor_resized / 255.0
        input_batch = tf.expand_dims(input_batch, 0)


        debug_img = (input_batch[0].numpy() * 255).astype(np.uint8)
        cv2.imwrite('debug_input.jpg', cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
        print(f"DEBUG: Zapisano debug_input.jpg. Sprawdź czy kolory są poprawne!")
        prediction = classifier.predict(input_batch, verbose=0)
        class_id = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        label_name = CLASSES.get(class_id, f"ID {class_id}")


        if confidence < MIN_CONFIDENCE:
            print(f"      -> CNN: Niepewny ({confidence:.1f}%)")
            continue
        
        # === SUKCES ===
        print(f"      -> [SUKCES] Klasa ID: {class_id} -> '{label_name}'")
        
        rect = plt.Rectangle((x1, y1), box_w, box_h, fill=False, color='#00FF00', linewidth=3)
        ax.add_patch(rect)
        
        display_text = f"{label_name} (ID:{class_id})"
        ax.text(x1, y1 - 10, display_text, color='yellow', fontsize=10, fontweight='bold', bbox=dict(facecolor='black', alpha=0.7))

    plt.axis('off')
    plt.title(f"Wynik Finalny\nPlik: {os.path.basename(IMAGE_PATH)}")
    
    # Zapis
    output_full_path = os.path.join(os.path.dirname(IMAGE_PATH), OUTPUT_FILENAME)
    print(f"\nZapisano wynik w: {output_full_path}")
    plt.savefig(output_full_path, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    run_pipeline()