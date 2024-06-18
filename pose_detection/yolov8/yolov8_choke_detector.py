import cv2
import numpy as np
from ultralytics import YOLO

# YOLOv8n Pose Model load
model = YOLO('yolov8n.pt')

# Define keypoint indices
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

def are_hands_near_neck(keypoints, threshold=50):
    """
    Checks if hands are near the neck
    :param keypoints: List of keypoints
    :param threshold: Distance threshold
    :return: Boolean indicating if hands are near the neck
    """
    neck = keypoints[KEYPOINTS["Neck"]]
    rwrist = keypoints[KEYPOINTS["RWrist"]]
    lwrist = keypoints[KEYPOINTS["LWrist"]]

    if neck is not None and rwrist is not None and lwrist is not None:
        dist_r = np.linalg.norm(np.array(neck[:2]) - np.array(rwrist[:2]))
        dist_l = np.linalg.norm(np.array(neck[:2]) - np.array(lwrist[:2]))

        if dist_r < threshold or dist_l < threshold:
            return True
    return False

# Path to video file
video_path = '/home/Steven/surveillance_system/pose_detection/yolov8/example_video.mp4'

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error opening video file {video_path}.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform pose estimation
    results = model(frame)

    # Iterate through the results
    for *box, conf, cls, kp in results.pred:  # Assuming 'kp' contains the keypoints
        # Check if hands are near the neck
        if are_hands_near_neck(kp):
            print("Hands are near the neck.")
            cv2.putText(frame, "Hands near neck!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Draw the detection on the frame
        cv2.rectangle(frame, box[:2], box[2:], (255, 0, 0), 2)

        cv2.imshow("Pose Estimation", frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
