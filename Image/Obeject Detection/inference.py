from ultralytics import YOLO
import cv2


model = YOLO("yolo11n.pt")  # Ensure this path is correct for your model

input_video_path = "path/to/your/video.mp4"  # Provide your input video path
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  # Run inference on the frame
    annotated_frame = results[0].plot()  # Plot detection results on the frame
    cv2.imshow('Video Inference', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
