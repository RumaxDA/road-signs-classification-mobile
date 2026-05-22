import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from app.core.config import settings 

class TrafficSignSystem:
    def __init__(self, yolo_path, cnn_path, img_size):
        print("--- Inicjalizacja systemów rozpoznawania ---")
        self.detector = YOLO(yolo_path)
        self.classifier = tf.keras.models.load_model(cnn_path)
        self.model = cnn_path
        self.img_size = img_size
        self.min_confidence = 0.3

        self.model_name = str(cnn_path).lower() if cnn_path else ""
        print(self.model_name)
        

        self.classes = {
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
    def preprocess_for_cnn(self, cropped_img):
        """Przygotowanie wyciętego znaku pod wejście CNN."""
        img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        
        is_efficientnet = "efficientnet" in self.model_name

        if not is_efficientnet:
            img = cv2.GaussianBlur(img, (3, 3), 0)
            
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
        
        if is_efficientnet:

            img = img.astype("float32") 
        else:
            img = img.astype("float32") / 255.0 
            
        return np.expand_dims(img, axis=0)

    def predict(self, frame):
        """Główna metoda przetwarzająca obraz."""
        results_list = []
        h, w, _ = frame.shape

        yolo_results = self.detector(frame, conf=0.2, iou=0.45, verbose=True)
        detections = yolo_results[0].boxes

        processed_boxes = []

        def get_iou(boxA, boxB):
            """Oblicza Intersection over Union (IoU) między dwiema ramkami [x1, y1, x2, y2]."""
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
            
            if float(boxAArea + boxBArea - interArea) == 0:
                return 0.0
            return interArea / float(boxAArea + boxBArea - interArea)
    
        sorted_indices = sorted(range(len(detections)), key=lambda i: detections.conf[i], reverse=True)

        for i in sorted_indices:
            box = detections[i]
            
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords

            orig_w, orig_h = x2 - x1, y2 - y1
            margin = 0.05
            
            dx = int(orig_w * margin)
            dy = int(orig_h * margin)
            
            x1 += dx
            y1 += dy
            x2 -= dx
            y2 -= dy

            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            current_box = [int(x1), int(y1), int(x2), int(y2)]
            
            box_w, box_h = x2 - x1, y2 - y1
            if box_h == 0 or box_w == 0: continue
            
            is_duplicate = False
            for p_box in processed_boxes:
                if get_iou(current_box, p_box) > 0.3:
                    is_duplicate = True
                    break
            
            if is_duplicate:
                print(f"DEBUG: Odrzucono powieloną ramkę: {current_box}")
                continue
                
            processed_boxes.append(current_box)

            print(f"DEBUG: YOLO znalazło obiekt: {x1, y1, x2, y2}")

            aspect_ratio = box_w / box_h
            if aspect_ratio < 0.5 or aspect_ratio > 1.5:
                continue

            cropped = frame[y1:y2, x1:x2]
            input_batch = self.preprocess_for_cnn(cropped)
            
            prediction = self.classifier.predict(input_batch, verbose=0)
            class_id = np.argmax(prediction)
            confidence = float(np.max(prediction))

            print(f"DEBUG KLASYFIKATORA: Obiekt {current_box} -> Klasa: {class_id}, Pewność: {confidence:.4f}")

            if confidence >= self.min_confidence:
                results_list.append({
                    "box": current_box,
                    "class_id": int(class_id),
                    "label": self.classes.get(int(class_id), f"ID {int(class_id)}"),
                    "confidence": round(float(confidence), 2)
                })
                
        return results_list

if __name__ == "__main__":
    YOLO_PATH = settings.YOLO_MODEL_PATH
    CNN_PATH = settings.CNN_MODEL_PATH
    TL_PATH = settings.TL_MODEL_PATH
    TL_96_PATH = settings.TL_MODEL_PATH
    
    system = TrafficSignSystem(YOLO_PATH, TL_PATH, 224)

    if "efficientnet" in system.model_name:
        print('preprocesing dla efficientneta') 
    else:
        print('/255 dla CNN') 
   
    
    test_img = cv2.imread('app/services/sign25.jpg')
    if test_img is not None:
        predictions = system.predict(test_img)
        print("\nWyniki rozpoznawania (JSON):")
        print(predictions)