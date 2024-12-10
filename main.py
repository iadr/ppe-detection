from sympy.printing.pretty.pretty_symbology import line_width
from ultralytics import YOLO

# Load a model

# target_model = "Construction-Site-Safety-PPE-Detection/models/best.pt"
target_model = "Construction-PPE-Detection/ppe.pt"
# target_model = "ppe_detection/yolov8s_custom.pt"
model = YOLO(target_model)  # pretrained YOLO11n model

# Run batched inference on a list of images
# results = model.predict("img2.jpg", save_crop=True, line_width=1)  # return a list of Results objects
results = model.predict("./images", save=True, stream=True)


def main():
    # Process results list
    for result in results:

        result.save(filename="result.jpg")  # save to disk


if __name__ == '__main__':
    main()
