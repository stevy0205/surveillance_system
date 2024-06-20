import os
from matplotlib import pyplot as plt
from ultralytics import YOLO
import cv2
import numpy as np
import time

# Load a model
#model = YOLO("yolov8n-pose.yaml")  # build a new model from YAML
#model = YOLO("yolov8n-pose.pt")  # load a pretrained model (recommended for training)
#model = YOLO("yolov8n-pose.yaml").load("yolov8n-pose.pt")  # build from YAML and transfer weights

# Train the model on coco128 pose
#results = model.train(data="coco8-pose.yaml", epochs=100, imgsz=640)
# Pfad zum aktuellen Verzeichnis der ausführenden Datei
current_dir = os.path.dirname(os.path.abspath(__file__))
# Relativer Pfad zum Modellgewicht
relative_path = "../../evals/yolov8_pose_detector_coco128_100ep/weights/best.pt"

# Absoluten Pfad zum Modellgewicht aufbauen
model_path = os.path.normpath(os.path.join(current_dir, relative_path))

model = YOLO(model_path)  # load a custom model

# Validate the model
#metrics = model.val()  # no arguments needed, dataset and settings remembered
#metrics.box.map  # map50-95
#metrics.box.map50  # map50
#metrics.box.map75  # map75
#metrics.box.maps  # a list contains map50-95 of each category

# Farben definieren
colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64)
]

# COCO Keypoint-Verbindungen
keypoint_connections = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Oberkörper
    (5, 6), (5, 7), (7, 9), (6, 8),  # Arme
    (10, 11), (5, 10), (6, 11),      # Hüfte und Beine
    (11, 13), (10, 12), (12, 14), (13, 15)  # Beine
]
# Keypoint-Klassen
keypoint_classes = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", 
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", 
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", 
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]
# Funktion zum Visualisieren der Keypoints und Verbindungen mit unterschiedlichen Farben
def plot_keypoints_connections(image, keypoints):
    for connection in keypoint_connections:
        start_point = tuple(keypoints[connection[0]].astype(int))
        end_point = tuple(keypoints[connection[1]].astype(int))
        if all(start_point) and all(end_point):  # Stellen Sie sicher, dass die Punkte gültig sind
            cv2.line(image, start_point, end_point, (0, 255, 255), 2)  # Gelbe Linien für Verbindungen
    for i, keypoint in enumerate(keypoints):
        x, y, confidence = keypoint
        if confidence > 0:  # Nur Keypoints mit einer Konfidenz größer als 0 anzeigen
            cv2.circle(image, (int(x), int(y)), 5, colors[i % len(colors)], -1)
    return image

# Funktion zur Berechnung der Euclidean-Distanz
def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


# Pfade zum Eingabe- und Ausgabedateien
#input_video_path = "pose_detection\yolov8_pose\WIN_20240616_22_14_05_Pro.mp4"
input_video_path = 0;
output_video_path = "pose_detection\yolov8_pose\output_video4.mp4"

# Überprüfung, ob der Eingabepfad existiert
if not os.path.exists(input_video_path):
    print(f"Fehler: Der Pfad '{input_video_path}' existiert nicht.")
    exit()

# Überprüfung, ob der Ausgabepfad existiert und schreibbar ist
if not os.path.exists(output_video_path) or not os.access(output_video_path, os.W_OK):
    print(f"Fehler: Der Pfad '{output_video_path}' existiert nicht oder ist nicht schreibbar.")
    exit()

# Video öffnen
cap = cv2.VideoCapture(input_video_path)

# Videoeigenschaften abrufen
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# VideoWriter für das Ausgabevideo einrichten
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))


# Fenster für die Anzeige der Pose öffnen
cv2.namedWindow('Pose Detection', cv2.WINDOW_NORMAL)

# Video frameweise verarbeiten
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

 # Vorhersage machen
    results = model(frame)

    # Variablen für die Keypoints initialisieren
    left_shoulder = (0,0,0)
    right_shoulder = (0,0,0)
    left_wrist =(0,0,0)
    right_wrist = (0,0,0)

 # Keypoints auf das Bild zeichnen und Koordinaten zwischenspeichern
    for result in results:
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints = result.keypoints.data.cpu().numpy()[0]  # Nehmen Sie die erste Pose an
            for i, keypoint in enumerate(keypoints):
                x, y, confidence = keypoint
                if confidence > 0.3:  # Nur Keypoints mit einer Konfidenz größer als 0 anzeigen
                    if i == 5:  # linke Schulter
                        left_shoulder = (x, y, confidence)
                    elif i == 6:  # rechte Schulter
                        right_shoulder = (x, y, confidence)
                    elif i == 9:  # linkes Handgelenk
                        left_wrist = (x, y, confidence)
                    elif i == 10:  # rechtes Handgelenk
                        right_wrist = (x, y, confidence)
                #if confidence > 0.35:
                    # Optional: Ausgabe der Koordinaten und Konfidenz
                    #print(f"{keypoint_classes[i]}: x={x}, y={y}, confidence={confidence}")
                    

            # Überprüfen, ob alle benötigten Keypoints erkannt wurden
            if (right_wrist is not None and right_shoulder is not None and
                left_wrist is not None and left_shoulder is not None):
                neck = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)
                neck_confidence = (left_shoulder[2] + right_shoulder[2]) / 2

                bias = 100  # Beispielhafter Schwellenwert für die Nähe
                hands_near_neck = False
                if (left_wrist is not None and euclidean_distance(left_wrist[:2], neck) < bias) or \
                   (right_wrist is not None and euclidean_distance(right_wrist[:2], neck) < bias):
                    hands_near_neck = True
                    print("Choke detected!")
                    
                if hands_near_neck:
                    if hands_near_neck_start_time is None:
                        hands_near_neck_start_time = time.time()
                    elif time.time() - hands_near_neck_start_time > 2:
                        print("Hands near neck detected for 2 seconds!")
                else:
                    hands_near_neck_start_time = None

            # Plot-Funktion außerhalb der Schleife aufrufen, um Verbindungen zwischen allen Keypoints zu zeichnen
            frame = plot_keypoints_connections(frame, keypoints)

    # Frame anzeigen
    cv2.imshow('Pose Detection', frame)

    # Press 'q' on the keyboard to exit the loop early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Ressourcen freigeben
cap.release()
cv2.destroyAllWindows()