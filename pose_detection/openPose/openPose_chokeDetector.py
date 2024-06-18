import cv2
import numpy as np
import os

# Pfad zu OpenPose
openpose_path = "/path/to/openpose/"

# Lade das OpenPose Modell
protoFile = openpose_path + "models/pose/coco/pose_deploy_linevec.prototxt"
weightsFile = openpose_path + "models/pose/coco/pose_iter_440000.caffemodel"

# Parameter für das Netz
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
inWidth = 368
inHeight = 368
threshold = 0.1

# Bild laden
image_path = "path/to/your/image.jpg"
frame = cv2.imread(image_path)
frameHeight, frameWidth = frame.shape[:2]

# Vorbereitung des Bildes für das Netz
inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
net.setInput(inpBlob)

# Vorhersage der Pose
output = net.forward()

# Definition der Keypoints, die wir extrahieren wollen
nPoints = 18
POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], 
              [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]]

# Keypoint Indexe für COCO Model
KEYPOINTS = {
    "Nose": 0,
    "Neck": 1,
    "RShoulder": 2,
    "RElbow": 3,
    "RWrist": 4,
    "LShoulder": 5,
    "LElbow": 6,
    "LWrist": 7,
    "RHip": 8,
    "RKnee": 9,
    "RAnkle": 10,
    "LHip": 11,
    "LKnee": 12,
    "LAnkle": 13,
    "REye": 14,
    "LEye": 15,
    "REar": 16,
    "LEar": 17
}

# Extrahieren der Keypoints
points = []
for i in range(nPoints):
    probMap = output[0, i, :, :]
    probMap = cv2.resize(probMap, (frameWidth, frameHeight))
    
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    
    if prob > threshold:
        points.append((int(point[0]), int(point[1])))
    else:
        points.append(None)

# Zeichnen der Keypoints und Verbindungen
for pair in POSE_PAIRS:
    partA = pair[0]
    partB = pair[1]

    if points[partA] and points[partB]:
        cv2.line(frame, points[partA], points[partB], (0, 255, 0), 2)
        cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

# Anzeigen des Ergebnisses
cv2.imshow("Output-Keypoints", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()


def are_hands_near_neck(keypoints, threshold=50):
    """
    Prüft, ob Hände in der Nähe des Halses sind
    :param keypoints: Liste der Keypoints
    :param threshold: Abstandsschwelle
    :return: Boolean, ob Hände in der Nähe des Halses sind
    """
    neck = keypoints[KEYPOINTS["Neck"]]
    rwrist = keypoints[KEYPOINTS["RWrist"]]
    lwrist = keypoints[KEYPOINTS["LWrist"]]
    
    if neck and rwrist and lwrist:
        dist_r = np.linalg.norm(np.array(neck) - np.array(rwrist))
        dist_l = np.linalg.norm(np.array(neck) - np.array(lwrist))
        
        if dist_r < threshold or dist_l < threshold:
            return True
    return False

if are_hands_near_neck(points):
    print("Hände befinden sich am Hals.")
else:
    print("Hände befinden sich nicht am Hals.")
