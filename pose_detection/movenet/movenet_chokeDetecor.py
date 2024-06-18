import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Laden Sie das MoveNet Modell
movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

# Öffnen Sie das lokale Video
video_path = '/home/Steven/surveillance_system/pose_detection/yolov8/example_video.mp4'
cap = cv2.VideoCapture(video_path)

while True:
    # Lesen Sie einen Frame aus dem Video
    ret, frame = cap.read()
    if not ret:
        break

    # Normalisieren Sie das Bild
    img = cv2.resize(frame, (192, 192))
    img = img.astype(np.float32) / 127.5 - 1

    # Führen Sie das Modell aus
    output = movenet(tf.constant(img[np.newaxis,...]))

    # Extrahieren Sie die Keypoints
    keypoints = output['output_0'].numpy()[0]

    # Filtern Sie die Keypoints für Hände und Hals
    hand_points = keypoints[[0, 4, 8, 12, 16]]  # Indexe für Hände
    neck_point = keypoints[1]  # Index für den Hals

    # Zeichnen Sie die Keypoints auf dem Bild
    for i, point in enumerate(hand_points):
        cv2.circle(frame, tuple(point[:2].astype(int)), 5, (0, 255, 0), -1)
    
    cv2.circle(frame, tuple(neck_point[:2].astype(int)), 5, (0, 0, 255), -1)
    
    # Zeigen Sie das Bild an
    cv2.imshow('Hand and Neck Detection', frame)
    
    # Brechen Sie die Schleife, wenn 'q' gedrückt wird
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
