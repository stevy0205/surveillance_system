import cv2
from cv2.gapi.wip.draw import Image
from sympy import python
from ultralytics import YOLO
from object_detection.yolov9 import val

# Build a YOLOv9c model from pretrained (COCO8 weight)
model = YOLO("yolov9c.pt")

# Train the model with imagenet dataset
#results = model.train(data="imagenet", epochs=100, imgsz=224)

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
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

# Your video processing loop remains unchanged
#video_path = r"D:\surveillance-system\data\ufcData\Anomaly-Videos-Part-1\Abuse\Abuse001_x264.mp4"
#cap = cv2.VideoCapture(video_path)
#while True:
#    success, img = cap.read()
#    if not success:
#        break
#    result_img, _ = predict_and_detect(model, img, classes=[], conf=0.5)
#    cv2.imshow("Image", result_img)
#    cv2.waitKey(1)


image = cv2.imread("D:\surveillance-system\data\knifeExample\knive2.jpeg")

result_img, _ = predict_and_detect(model, image, classes=[43], conf=0.5)

cv2.imwrite("YourSavePath", result_img)

window_size = (768, 600)  # Doppelte Breite und Höhe des Originalbildes
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", window_size[0], window_size[1])
cv2.imshow("Image", result_img)

# Warten auf eine Taste gedrückt zu bekommen
cv2.waitKey(0)
cv2.destroyAllWindows()
