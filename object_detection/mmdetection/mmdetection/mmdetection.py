import cv2
from mmdet.apis import DetInferencer

# Initialize the Detector
detector = DetInferencer('rtmdet_x_8xb32-300e_coco')

# Video path
video_path = "C:Users/steve/IdeaProjects/surveillance_system_vscode_win/surveillance_system/data/recordings/knife/bullet/p1_24_06_1748.mp4"  # 0 refers to the default camera

# Open the video stream
cap = cv2.VideoCapture(video_path)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("No image captured.")
        break

    # Process the frame with the detector
    detector(frame, show=True)

    # Display the processed frame
    cv2.imshow("Video", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# Release the VideoCapture object and destroy all windows
cap.release()
cv2.destroyAllWindows()
