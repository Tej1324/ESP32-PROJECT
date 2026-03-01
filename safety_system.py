import cv2
from ultralytics import YOLO
import time

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

clear_delay = 3
last_person_time = None
machine_running = True

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Define danger zone (center box)
    zone_x1 = int(w * 0.3)
    zone_y1 = int(h * 0.3)
    zone_x2 = int(w * 0.7)
    zone_y2 = int(h * 0.8)

    results = model(frame)

    person_in_zone = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if model.names[cls] == "person":

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)

                # Check overlap with danger zone
                if (x1 < zone_x2 and x2 > zone_x1 and
                    y1 < zone_y2 and y2 > zone_y1):
                    person_in_zone = True

    current_time = time.time()

    if person_in_zone:
        machine_running = False
        last_person_time = current_time
    else:
        if last_person_time and current_time - last_person_time > clear_delay:
            machine_running = True

    # Draw danger zone
    cv2.rectangle(frame, (zone_x1, zone_y1),
                  (zone_x2, zone_y2), (0,0,255), 2)

    if machine_running:
        status = "MACHINE RUNNING"
        color = (0,255,0)
    else:
        status = "MACHINE STOPPED"
        color = (0,0,255)

    cv2.putText(frame, status, (40,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.imshow("Industrial Safety Prototype", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()