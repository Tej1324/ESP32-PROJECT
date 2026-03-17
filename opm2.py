from ultralytics import YOLO
import cv2
import time

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# RTSP camera URL
rtsp_url = "rtsp://admin:cctv%40999@192.168.0.147:554/stream1"

def connect_camera():
    print("Connecting to camera...")
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    if not cap.isOpened():
        print("Camera connection failed. Retrying in 3 seconds...")
        time.sleep(3)
        return connect_camera()

    print("Camera connected")
    return cap


cap = connect_camera()

frame_count = 0

while True:

    ret, frame = cap.read()

    # Reconnect if frame lost
    if not ret:
        print("Frame lost. Reconnecting...")
        cap.release()
        time.sleep(1)
        cap = connect_camera()
        continue

    frame_count += 1

    # Process every 3rd frame (reduces CPU usage)
    if frame_count % 3 != 0:
        continue

    # Resize frame for faster inference
    frame = cv2.resize(frame, (640, 480))

    # Run YOLO detection
    results = model(frame, conf=0.5)

    intrusion = False

    for r in results:
        for box in r.boxes:

            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "person":

                intrusion = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                # Label
                cv2.putText(frame,
                            "PERSON",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0,255,0),
                            2)

    if intrusion:
        print("PERSON DETECTED")

    # Display stream
    cv2.imshow("RTSP Human Detection", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()