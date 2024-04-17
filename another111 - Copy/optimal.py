import cv2
import numpy as np
import os
import threading
import time

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = ['yolo_82', 'yolo_94', 'yolo_106']  # Get output layers

# Specify the directory to store output files
output_dir = "./outputs/"

# Lock for synchronization
lock = threading.Lock()

# Flag to control detection
pause_detection = False

# Define the desired classes
desired_classes = ["fork", "cup", "bottle", "spoon"]

# Function for object detection
def detect_objects():
    global pause_detection
    while True:
        if not pause_detection:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (800, 600))
            print("Starting object detection...")
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            with lock:
                outs = net.forward(output_layers)
            detected_objects = set()
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        class_name = classes[class_id]
                        if class_name in desired_classes:  # Check if the detected class is in the desired classes
                            detected_objects.add(class_name)
            output_file_path = os.path.join(output_dir, "match.txt")
            with open(output_file_path, "w") as f:
                for obj in detected_objects:
                    f.write(obj + "\n")
            print("Object detection complete.")
            time.sleep(0.1)  # Optional delay to reduce CPU usage

# Function to check for existing output file
def check_output_file():
    global pause_detection
    while True:
        existing_files = [file for file in os.listdir(output_dir) if file.endswith(".txt")]
        if existing_files:
            print("Existing files detected. Pausing detection until files are removed...")
            pause_detection = True
        else:
            print("No existing files found. Resuming detection.")
            pause_detection = False
        time.sleep(1)  # Check every second

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 for default webcam, change it if needed

# Start continuous output file checking thread
output_file_checking_thread = threading.Thread(target=check_output_file)
output_file_checking_thread.daemon = True  # Daemonize the thread so it exits when the main program exits
output_file_checking_thread.start()

# Start object detection thread
detection_thread = threading.Thread(target=detect_objects)
detection_thread.daemon = True
detection_thread.start()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
