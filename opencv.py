import cv2
from ultralytics import YOLO


target_model = "ppe_detection/yolov8s_custom.pt"
model = YOLO(target_model)  # pretrained YOLO11n model
image = cv2.imread("images/img7.jpg")
results = model(image, verbose=False)
classes = []
safety = ['Person', 'Glass', 'Gloves', 'Helmet', 'Safety-Vest', 'helmet']


def main():
    for r in results:
        for c in r.boxes:
            if model.names[int(c.cls)] in safety:
                class_name = model.names[int(c.cls)]
                classes.append(model.names[int(c.cls)])
                x1, y1, x2, y2 = map(int, c.xyxy[0])
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    cv2.imwrite('result.jpg', image)


if __name__ == "__main__":
    main()
