import glob
import os
import cv2
from cv2.gapi.wip.draw import Image
from sympy import python
from ultralytics import YOLO
import matplotlib.pyplot as plt
import torch

  
def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf,verbose=False)
    else:
        results = chosen_model.predict(img, conf=conf,verbose=False)
    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), text_thickness)
    return img, results

# Load Knife Model
model = YOLO("C:/Users/steve/IdeaProjects/surveillance_system_vscode_win/surveillance_system/evals/result_knifeDetector_set_100ep/weights/best.pt")

video_path = 0
# Stream
cap = cv2.VideoCapture(f'rtsp://stream:KI@JVA123@141.21.39.122:554/axis-media/media.amp?videocodec=h264&resolution=800x600')
while True:
    success, img = cap.read()
    if not success:
        break
    
    # Lesen des Live-Streams
    ret, frame = cap.read()
    height, width, layers = frame.shape
    frame = cv2.resize(frame, (width // 2, height // 2))
    result_img, _ = predict_and_detect(model, img, classes=[], conf=0.379)
    # Anzeigen des Video-Frames
    cv2.imshow("RTSP Camera Stream", frame)

    # Press 'q' on the keyboard to exit the loop early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.waitKey(1)
    # Hauptfunktion
if __name__ == "__main__":
    # Freigabe und Schließen des Streams
    cap.release()
    cv2.destroyAllWindows()
# Video path
#cap = cv2.VideoCapture(video_path)
#while True:
#    success, img = cap.read()
#    if not success:
#      break
 #   result_img, _ = predict_and_detect(model, img, classes=[], conf=0.379)
  #  cv2.imshow("Image", result_img)
   # 
    ## Press 'q' on the keyboard to exit the loop early
    #if cv2.waitKey(1) & 0xFF == ord('q'):
     #   break
    #cv2.waitKey(1)
    