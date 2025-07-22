import numpy as np
import cv2


def get_yolo_detections_and_annotated_image(
        frame, model, conf_thresh=0.3, face_class=0, show_label=True
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

    model = model
    results = model.predict(img, verbose=False)

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
                    label = f"{conf[i]:.2f}"
                    cv2.putText(
                        img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                    )
    return boxes, img
