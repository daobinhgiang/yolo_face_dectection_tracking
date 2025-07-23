from ultralytics import YOLO
import numpy as np
import pandas as pd
import cv2

from face_tracking_api.kf_tracking import best_model
from kf_tracking import MultiObjectTracker
from pred_annot_img import get_yolo_detections_and_annotated_image


def track_faces_and_save_video(
        input_video_path,
        output_video_path,
        yolo_model,
        face_class=0,
        conf_thresh=0.4,
        max_disappeared=60,
        iou_threshold=0.3
):
    # Initialize video capture and get video properties
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Calculate maximum frames for 20 seconds
    max_frames = int(fps * 10)  # 20 seconds worth of frames
    print(f"Processing first 20 seconds ({max_frames} frames) of video at {fps} FPS")

    # Initialize tracker
    tracker = MultiObjectTracker(max_disappeared=max_disappeared, iou_threshold=iou_threshold)

    # Function to convert YOLO bbox to tracker bbox
    def xyxy_to_xywh(box):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        x_center = x1 + w / 2
        y_center = y1 + h / 2
        return [x_center, y_center, w, h]

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"End of video reached at frame {frame_num}")
            break

        # Stop after 20 seconds
        if frame_num >= max_frames:
            print(f"Reached 20 second limit at frame {frame_num}")
            break

        # YOLO detection
        boxes_xyxy, *_ = get_yolo_detections_and_annotated_image(
            frame, yolo_model, conf_thresh=conf_thresh, face_class=face_class, show_label=False
        )

        detections = [xyxy_to_xywh(box) for box in boxes_xyxy]

        # Tracking
        track_results = tracker.update(detections, frame)

        # Draw tracking results
        for result in track_results:
            x, y, w, h = result['bbox']
            track_id = result['id']
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Fetch name/info from tracker
            tracker_obj = tracker.trackers[track_id]
            if tracker_obj.person_info is not None:
                label = tracker_obj.person_info.get("Name")
            else:
                label = f"Unknown"

            # print (label)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Write frame to output video
        out.write(frame)
        frame_num += 1

        cv2.imshow("temp", frame)
        cv2.waitKey(1)

        # Progress update every 100 frames
        if frame_num % 100 == 0:
            elapsed_seconds = frame_num / fps
            print(f"Processed {frame_num} frames ({elapsed_seconds:.1f} seconds)...")

    # Clean up
    cap.release()
    out.release()

    final_seconds = frame_num / fps
    print(f"Tracking complete. Processed {frame_num} frames ({final_seconds:.1f} seconds)")
    print(f"Output saved to {output_video_path}")


best_model = YOLO("/home/giangdb/Documents/ETC/face_tracking_api/best.pt")
track_faces_and_save_video(
    input_video_path="/home/giangdb/Documents/ETC/face_tracking_api/People Walking Free Stock Footage, Royalty-Free No Copyright Content(1).mp4",
    output_video_path="output1.mp4",
    yolo_model=best_model,
    face_class=0,  # change if your face class index differs
    conf_thresh=0.4,  # detection confidence threshold
    max_disappeared=1000,  # how many frames to keep lost tracks
    iou_threshold=0.3  # IOU threshold for matching
)
