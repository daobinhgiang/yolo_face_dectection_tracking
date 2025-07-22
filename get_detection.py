import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO


def get_yolo_detections_and_annotated_image(
        frame, model_path, conf_thresh=0.3, face_class=0, show_label=True
):
    """
    Detect objects and draw bounding boxes on the image.

    Args:
        frame: image file path (str) or np.ndarray (BGR or RGB).
        model_path: path to YOLO weights.
        conf_thresh: min confidence to keep detection.
        face_class: class index to filter.
        show_label: whether to display confidence and class on the box.
    Returns:
        boxes: [[x1, y1, x2, y2], ...]
        annotated_img: image (np.ndarray) with boxes drawn.
    """
    # Load image if path is given
    if isinstance(frame, str):
        img = cv2.imread(frame)
        if img is None:
            raise ValueError(f"Cannot read image at {frame}")
    else:
        img = frame.copy()

    # If image is grayscale, convert to 3 channels
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    model = YOLO(model_path)
    results = model.predict(img)

    boxes = []
    for r in results:
        b = r.boxes
        if b is None or b.xyxy is None:
            continue
        xyxy = b.xyxy.cpu().numpy()
        conf = b.conf.cpu().numpy()
        cls = b.cls.cpu().numpy()
        for i in range(len(xyxy)):
            if conf[i] >= conf_thresh and int(cls[i]) == face_class:
                x1, y1, x2, y2 = map(int, xyxy[i])
                boxes.append([x1, y1, x2, y2])
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if show_label:
                    label = f"Face {conf[i]:.2f}"
                    cv2.putText(
                        img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                    )
    return boxes, img


def show_annotated_img(img):
    annotated_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(annotated_img_rgb)
    plt.axis('off')
    plt.show()


model_path = "model_path"
boxes, annotated_img = get_yolo_detections_and_annotated_image(
    # frame="/kaggle/input/face-detection-dataset/images/train/001065b2a612f5ad.jpg",
    frame="video.mp4",
    model_path=model_path,
    conf_thresh=0.3,
    face_class=0,
    show_label=True
)

show_annotated_img(annotated_img)
