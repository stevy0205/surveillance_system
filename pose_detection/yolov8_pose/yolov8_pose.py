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

model = YOLO("/home/Steven/runs/pose/train/weights/best.pt")  # load a custom model

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list contains map50-95 of each category



# Predict with the model
results = model("/home/Steven/surveillance_system/pose_detection/yolov8_pose/WIN_20240619_13_44_03_Pro.jpg")  # predict on an image

# Verzeichnis zum Speichern der Ergebnisse
save_dir = "/home/Steven/surveillance_system/pose_detection/yolov8_pose/results"
os.makedirs(save_dir, exist_ok=True)

# Farben definieren
colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64)
]

# Funktion zum Visualisieren der Keypoints mit unterschiedlichen Farben
def plot_keypoints(image, keypoints):
    for i, keypoint in enumerate(keypoints):
        x, y, confidence = keypoint
        if confidence > 0:  # Nur Keypoints mit einer Konfidenz größer als 0 anzeigen
            cv2.circle(image, (int(x), int(y)), 5, colors[i % len(colors)], -1)
    return image


# Pfade zum Eingabe- und Ausgabedateien
input_video_path = "/home/Steven/surveillance_system/pose_detection/yolov8_pose/WIN_20240616_22_14_05_Pro.mp4"
output_video_path = "/home/Steven/surveillance_system/pose_detection/yolov8_pose/output_video.mp4"

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
            frame = plot_keypoints(frame, keypoints)

    # Frame ins Ausgabevideo schreiben
    out.write(frame)

# Ressourcen freigeben
cap.release()
out.release()


