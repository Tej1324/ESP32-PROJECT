from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# RTSP camera URL
rtsp_url = "rtsp://admin:cctv%40999@192.168.0.178:554/stream1"

# Open RTSP stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Could not open RTSP stream")
    exit()

while True:

    ret, frame = cap.read()
    if not ret:
     print("Frame lost... reconnecting")
     cap.release()
     cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
     continue

    # Resize for faster inference
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
                cv2.putText(frame, "PERSON",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0,255,0),
                            2)

    if intrusion:
        print("PERSON DETECTED")

    # Show output
    cv2.imshow("RTSP Human Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:   # ESC key
        break

cap.release()
cv2.destroyAllWindows()