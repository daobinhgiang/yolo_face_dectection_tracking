from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

import numpy as np
import cv2

best_model=YOLO('/home/giangdb/Documents/ETC/face_tracking_api/best.pt')


class KalmanTracker:
    def __init__(self, bbox, track_id):
        """
        bbox: [x_center, y_center, width, height]
        """
        self.track_id = track_id
        self.age = 0
        self.hits = 0
        self.time_since_update = 0

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
        self.trackers = []
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

    def associate_detections_to_trackers(self, detections, predictions):
        """Associate detections to existing trackers using IoU"""
        if len(predictions) == 0:
            return [], list(range(len(detections))), []

        if len(detections) == 0:
            return [], [], list(range(len(predictions)))

        # Calculate IoU matrix
        iou_matrix = np.zeros((len(detections), len(predictions)))
        for d, det in enumerate(detections):
            for t, pred in enumerate(predictions):
                iou_matrix[d, t] = self.calculate_iou(det, pred)

        # Use Hungarian algorithm for optimal assignment
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > self.iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = linear_sum_assignment(-iou_matrix)
                matched_indices = np.array(list(zip(*matched_indices)))
        else:
            matched_indices = np.empty(shape=(0, 2))

        # Filter out matched pairs with low IoU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                continue
            matches.append(m.reshape(2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.array(matches)

        # Identify unmatched detections and trackers
        unmatched_detections = []
        for d, det in enumerate(detections):
            if len(matches) == 0 or d not in matches[:, 0]:
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t, trk in enumerate(self.trackers):
            if len(matches) == 0 or t not in matches[:, 1]:
                unmatched_trackers.append(t)

        return matches, unmatched_detections, unmatched_trackers

    def update(self, detections):
        """
        Update trackers with new detections
        detections: list of bounding boxes [[x_center, y_center, width, height], ...]
        """
        # Predict next positions for all trackers
        predictions = []
        for tracker in self.trackers:
            pred = tracker.predict()
            predictions.append(pred)

        # Associate detections to trackers
        matches, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(
            detections, predictions
        )

        # Update matched trackers with assigned detections
        for m in matches:
            self.trackers[m[1]].update(detections[m[0]])

        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            tracker = KalmanTracker(detections[i], self.next_id)
            self.trackers.append(tracker)
            self.next_id += 1

        # Remove dead trackers
        self.trackers = [
            tracker for tracker in self.trackers
            if tracker.time_since_update <= self.max_disappeared
        ]

        # Return current tracking results
        results = []
        for tracker in self.trackers:
            if tracker.time_since_update <= 1:  # Only return recent tracks
                bbox = tracker.get_state()
                results.append({
                    'id': tracker.track_id,
                    'bbox': bbox,
                    'hits': tracker.hits,
                    'age': tracker.age
                })

        return results
