import cv2
from ultralytics import YOLO
import time

# Load lightweight YOLO model
model = YOLO("yolov8n.pt")

# Use Mac built-in webcam
cap = cv2.VideoCapture(0)

clear_delay = 3  # seconds required before restart
last_person_time = None
machine_running = True

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    person_detected = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if model.names[cls] == "person":
                person_detected = True

    current_time = time.time()

    if person_detected:
        machine_running = False
        last_person_time = current_time
    else:
        if last_person_time and current_time - last_person_time > clear_delay:
            machine_running = True

    if machine_running:
        status = "MACHINE RUNNING"
        color = (0, 255, 0)
    else:
        status = "MACHINE STOPPED"
        color = (0, 0, 255)

    cv2.putText(frame, status, (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 3)

    cv2.imshow("Industrial Safety Prototype", frame)

    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()