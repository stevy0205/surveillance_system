import os
from matplotlib import pyplot as plt
from ultralytics import YOLO
import cv2

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
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list contains map50-95 of each category

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


# Pfade zum Eingabe- und Ausgabedateien
input_video_path = "../surveillance_system/pose_detection/yolov8_pose/WIN_20240616_22_14_05_Pro.mp4"
output_video_path = "../surveillance_system/pose_detection/yolov8_pose/output_video3.mp4"

# Video öffnen
cap = cv2.VideoCapture(input_video_path)

# Videoeigenschaften abrufen
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# VideoWriter für das Ausgabevideo einrichten
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Video frameweise verarbeiten
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Vorhersage machen
    results = model(frame)

    # Keypoints auf das Bild zeichnen
    for result in results:
        if hasattr(result, 'keypoints') and result.keypoints is not None:
            keypoints = result.keypoints.data.cpu().numpy()[0]  # Nehmen Sie die erste Pose an
            frame = plot_keypoints_connections(frame, keypoints)

    # Frame ins Ausgabevideo schreiben
    out.write(frame)

# Ressourcen freigeben
cap.release()
out.release()


