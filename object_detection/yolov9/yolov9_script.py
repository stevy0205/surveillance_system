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
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)
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

# Build a YOLOv9c model from pretrained (COCO8 weight)
#model = YOLO("yolov9c.pt")

#Load Hands and face model
#model = YOLO("C:/Users/steve/IdeaProjects/surveillance_system_vscode_win/surveillance_system/evals/hand_and_face_results_100ep/weights/best.pt")

# Load Rope Model
#model = YOLO("C:/Users/steve/IdeaProjects/surveillance_system_vscode_win/surveillance_system/evals/hang_and_rope_dataset_100ep/weights/best.pt")

# Load Knife Model
model = YOLO("C:/Users/steve/IdeaProjects/surveillance_system_vscode_win/surveillance_system/evals/result_knifeDetector_set_100ep/weights/best.pt")

# Train the model on the COCO8 example dataset for 100 epochs
#results = model.train(data="coco8.yaml", epochs=100, imgsz=224)

# Train the model on the knife dataset (https://universe.roboflow.com/sadi0v2/knife-dataset-lfxmz/dataset/4)
#results = model.train(data="/home/Steven/surveillance_system/object_detection/yolov9/knife_dataset/data.yaml", epochs=100, imgsz=224)

# Train the model on the knife dataset (https://universe.roboflow.com/ai-0jtbr/knife-detection-hgvy2/dataset/2)
#results = model.train(data="/home/Steven/surveillance_system/object_detection/yolov9/knife-detection Image Dataset/data.yaml", epochs=100, imgsz=224)

# Train the model on the rope dataset (https://universe.roboflow.com/ningbo-university/rope_real/dataset/2#)
#results = model.train(data="/home/Steven/surveillance_system/object_detection/yolov9/rope_dataset/data.yaml", epochs=100, imgsz=224)

# Train the model on the hang and rope dataset with aug (https://universe.roboflow.com/test-pgkqh/hang-rope/dataset/1)
#results = model.train(data="/home/Steven/surveillance_system/datasets/hang_and_rope_dataset/data.yaml", epochs=100, imgsz=224)

# Display model information (optional)
#model.info()

#metrics = model.val(data="/home/Steven/surveillance_system/object_detection/yolov9/custom-coco.yaml")
#person_metrics = metrics.box.class_result(43) # Obtain metrics about the `Person` class (i.e., class 0)

# Your video processing loop remains unchanged
#video_path = "C:\Users\steve\IdeaProjects\surveillance_system_vscode_win\surveillance_system\pose_detection\yolov8_pose\WIN_20240616_22_14_05_Pro.mp4"
video_path = 0
cap = cv2.VideoCapture(video_path)
while True:
    success, img = cap.read()
    if not success:
        break
    result_img, _ = predict_and_detect(model, img, classes=[], conf=0.379)
    cv2.imshow("Image", result_img)
    
    # Press 'q' on the keyboard to exit the loop early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.waitKey(1)
    

#image = cv2.imread("surveillance_system/data/knifeExample/knife2.jpeg")

#result_img, _ = predict_and_detect(model, image, classes=[0], conf=0.5)


#cv2.imshow("Image",result_img)
# Speichern Sie das Ergebnisbild
#cv2.imwrite("surveillance_system/data/knifeExample/knife2_res.jpeg", result_img)

#window_size = (768, 600)  # Doppelte Breite und Höhe des Originalbildes
#cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
#cv2.resizeWindow("Image", window_size[0], window_size[1])
#cv2.imshow("Image", result_img)

# Warten auf eine Taste gedrückt zu bekommen
#cv2.waitKey(0)
#cv2.destroyAllWindows()

