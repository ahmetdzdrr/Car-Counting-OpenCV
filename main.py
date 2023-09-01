from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("C:/Users/ahmet/PycharmProjects/CarCounting/videos/cars.mp4")
model = YOLO("C:/Users/ahmet/PycharmProjects/CarCounting/weights/yolov8n.pt")
mask = cv2.imread("images/mask.png")
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits = [380, 320, 700, 320]
total_count = []

label_class = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
               "teddy bear", "hair drier", "toothbrush"
               ]

while True:
    success, img = cap.read()
    masked = cv2.bitwise_and(img, mask)

    results = model(masked, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            current_class = label_class[cls]

            if current_class in ["car", "truck", "bus"] and conf > 0.4:
                cvzone.putTextRect(img, f'{current_class}-{conf}', (max(0, x1), max(0, y1)),
                                   scale=1.2, thickness=1, offset=3, colorR=(255, 0, 255))
                current_list = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_list))

    tracker_result = tracker.update(detections)

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    for result in tracker_result:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if total_count.count(id) == 0:
                total_count.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]),
                         (0, 255, 0), 5)

    cv2.putText(img, f'COUNT: {str(len(total_count))}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                5, (255, 255, 255), 16, cv2.LINE_AA)

    cv2.imshow("Car Counter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
