import time

from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from name_process import crop_and_encode_face, get_face_embedding, get_person_info
import ast
import numpy as np
import cv2

best_model = YOLO('/home/giangdb/Documents/ETC/face_tracking_api/best.pt')


class KalmanTracker:
    def __init__(self, bbox, track_id):
        """
        bbox: [x_center, y_center, width, height]
        """
        self.track_id = track_id
        self.age = 0
        self.hits = 0
        self.time_since_update = 0
        self.person_info = None  # To store person's info

        # State vector: [x, y, w, h, vx, vy, vw, vh]
        # Position and velocity for center coordinates and dimensions
        self.kf = cv2.KalmanFilter(8, 4)
        # State transition matrix (constant velocity model)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + vx
            [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + vy
            [0, 0, 1, 0, 0, 0, 1, 0],  # w = w + vw
            [0, 0, 0, 1, 0, 0, 0, 1],  # h = h + vh
            [0, 0, 0, 0, 1, 0, 0, 0],  # vx = vx
            [0, 0, 0, 0, 0, 1, 0, 0],  # vy = vy
            [0, 0, 0, 0, 0, 0, 1, 0],  # vw = vw
            [0, 0, 0, 0, 0, 0, 0, 1]  # vh = vh
        ], dtype=np.float32)

        # Measurement matrix (we observe x, y, w, h)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)

        # Process noise covariance (how much we trust the model)
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32)
        self.kf.processNoiseCov[4:, 4:] *= 0.01  # Lower noise for velocities

        # Measurement noise covariance (how much we trust measurements)
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1

        # Error covariance matrix
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)

        # Initialize state
        self.kf.statePre = np.array([bbox[0], bbox[1], bbox[2], bbox[3], 0, 0, 0, 0], dtype=np.float32)
        self.kf.statePost = self.kf.statePre.copy()

    def predict(self):
        """Predict the next state using Kalman filter"""
        # Increment age and time since update
        self.age += 1
        self.time_since_update += 1

        # Predict next state
        predicted_state = self.kf.predict()

        # Return predicted bounding box [x_center, y_center, width, height]
        return predicted_state[:4].flatten()

    def set_info(self, info):
        self.person_info = info

    def update(self, bbox):
        """Update the Kalman filter with a new measurement"""
        self.time_since_update = 0
        self.hits += 1

        # Convert bbox to measurement vector
        measurement = np.array([bbox[0], bbox[1], bbox[2], bbox[3]], dtype=np.float32)

        # Update the filter
        self.kf.correct(measurement)

    def get_state(self):
        """Get current state as bounding box"""
        return self.kf.statePost[:4].flatten()


class MultiObjectTracker:
    def __init__(self, max_disappeared=20, iou_threshold=0.3):
        self.next_id = 0
        self.trackers = {}
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) of two bounding boxes"""

        # Convert center format to corner format
        def center_to_corner(box):
            x, y, w, h = box
            return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]

        box1_corner = center_to_corner(box1)
        box2_corner = center_to_corner(box2)

        # Calculate intersection
        x1 = max(box1_corner[0], box2_corner[0])
        y1 = max(box1_corner[1], box2_corner[1])
        x2 = min(box1_corner[2], box2_corner[2])
        y2 = min(box1_corner[3], box2_corner[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # Calculate union
        area1 = (box1_corner[2] - box1_corner[0]) * (box1_corner[3] - box1_corner[1])
        area2 = (box2_corner[2] - box2_corner[0]) * (box2_corner[3] - box2_corner[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def associate_detections_to_trackers(self, detections, predictions, tracker_ids):
        """
        Modified: Accepts tracker_ids so we can match detection <-> tracker_id robustly.
        """
        if len(predictions) == 0:
            return [], list(range(len(detections))), []

        if len(detections) == 0:
            return [], [], list(tracker_ids)

        iou_matrix = np.zeros((len(detections), len(predictions)))
        for d, det in enumerate(detections):
            for t, pred in enumerate(predictions):
                iou_matrix[d, t] = self.calculate_iou(det, pred)

        matched_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = np.array(list(zip(*matched_indices)))

        # Filter out low IoU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] >= self.iou_threshold:
                matches.append((m[0], m[1]))

        matched_det_indices = set(m[0] for m in matches)
        matched_trk_indices = set(m[1] for m in matches)

        unmatched_detections = [i for i in range(len(detections)) if i not in matched_det_indices]
        unmatched_tracker_ids = [tracker_ids[i] for i in range(len(predictions)) if i not in matched_trk_indices]

        return matches, unmatched_detections, unmatched_tracker_ids

    def update(self, detections, frame):
        """
        Update trackers with new detections.
        Returns: list of dicts, each {'id', 'bbox', 'hits', 'age'}
        """
        tracker_ids = list(self.trackers.keys())
        predictions = [self.trackers[tid].predict() for tid in tracker_ids]

        matches, unmatched_dets, unmatched_trk_ids = self.associate_detections_to_trackers(
            detections, predictions, tracker_ids
        )

        # Update matched trackers
        for det_idx, trk_idx in matches:
            track_id = tracker_ids[trk_idx]
            self.trackers[track_id].update(detections[det_idx])

        # Increment time_since_update for unmatched trackers
        for track_id in unmatched_trk_ids:
            self.trackers[track_id].time_since_update += 1

        # Create new trackers for unmatched detections
        for det_idx in unmatched_dets:
            tracker = KalmanTracker(detections[det_idx], self.next_id)
            self.trackers[self.next_id] = tracker
            self.next_id += 1

        # Remove trackers that have disappeared for too long
        to_remove = [tid for tid, tracker in self.trackers.items() if tracker.time_since_update > self.max_disappeared]
        for tid in to_remove:
            del self.trackers[tid]

        # Return results for all active trackers
        results = []
        for track_id, tracker in self.trackers.items():
            if tracker.time_since_update <= 1:
                bbox = tracker.get_state()
                if tracker.person_info is None:
                    base64_img = crop_and_encode_face(frame, bbox)
                    start = time.time()
                    embedding = get_face_embedding(base64_img)
                    info = get_person_info(embedding)
                    end = time.time()
                    print(end - start)
                    tracker.set_info(info)
                # if info is not None:
                #     tracker.set_info(info)
                results.append({
                    'id': track_id,
                    'bbox': bbox,
                    'hits': tracker.hits,
                    'age': tracker.age
                })

        return results
