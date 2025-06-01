import numpy as np
from collections import deque
import torch
import cv2
from scipy.optimize import linear_sum_assignment


class KalmanFilter:
    """
    A simple Kalman filter implementation for object tracking.
    State vector: [x, y, a, h, vx, vy, va, vh]
    where (x, y) is the center position, a is the aspect ratio, h is the height,
    and (vx, vy, va, vh) are the respective velocities.
    """
    def __init__(self):
        self.motion_mat = np.eye(8, 8)
        # dt for position prediction
        self.motion_mat[0, 4] = 1
        self.motion_mat[1, 5] = 1
        self.motion_mat[2, 6] = 1
        self.motion_mat[3, 7] = 1
        
        self.measurement_mat = np.eye(4, 8)  # Measurement matrix (4x8)
        
        # Process noise covariance
        self.process_noise = np.eye(8) * 0.01
        
        # Measurement noise covariance
        self.measurement_noise = np.eye(4) * 0.1
        
    def predict(self, state, covariance):
        """Predict the next state and covariance based on motion model."""
        # Predicted state
        predicted_state = self.motion_mat @ state
        
        # Predicted covariance
        predicted_covariance = self.motion_mat @ covariance @ self.motion_mat.T + self.process_noise
        
        return predicted_state, predicted_covariance
    
    def update(self, state, covariance, measurement):
        """Update the state and covariance based on measurement."""
        # Innovation/residual
        innovation = measurement - self.measurement_mat @ state
        
        # Innovation covariance
        innovation_cov = self.measurement_mat @ covariance @ self.measurement_mat.T + self.measurement_noise
        
        # Kalman gain
        kalman_gain = covariance @ self.measurement_mat.T @ np.linalg.inv(innovation_cov)
        
        # Updated state
        updated_state = state + kalman_gain @ innovation
        
        # Updated covariance
        identity = np.eye(8)
        updated_covariance = (identity - kalman_gain @ self.measurement_mat) @ covariance
        
        return updated_state, updated_covariance


class STrack:
    """Single Track class for ByteTrack."""
    _next_id = 1  # Class variable for ID generation
    
    def __init__(self, bbox, confidence, feature=None):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.feature = feature
        self.track_id = 0  # Will be assigned when confirmed
        self.state = 'new'  # new, tracked, lost
        self.age = 0
        self.hits = 0
        self.time_since_update = 0
        
        # Initialize Kalman filter state
        self.kalman_filter = KalmanFilter()
        
        # Initialize kalman state and covariance
        self.mean, self.covariance = self._init_kalman_state(bbox)
        
        # Tracking history
        self.history = deque(maxlen=100)  # Increased history size
        
        # Additional attributes for insect tracking
        self.insect_class = None
        self.insect_type = None
        
    def _init_kalman_state(self, bbox):
        """Initialize the Kalman filter state from a bounding box."""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        aspect_ratio = w / h
        
        # Initial state [x, y, a, h, vx, vy, va, vh]
        mean = np.array([cx, cy, aspect_ratio, h, 0, 0, 0, 0], dtype=np.float32)
        
        # Initial covariance
        covariance = np.eye(8, dtype=np.float32) * 10
        
        return mean, covariance
    
    def predict(self):
        """Predict the next state using Kalman filter."""
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
        
    def update(self, detection):
        """Update state with new detection."""
        self.bbox = detection.bbox
        self.confidence = detection.confidence
        if detection.feature is not None:
            self.feature = detection.feature
            
        # Convert bbox to measurement [cx, cy, aspect_ratio, height]
        measurement = self._bbox_to_measurement(detection.bbox)
        
        # Update Kalman filter
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, measurement)
        
        # Update tracking status
        self.hits += 1
        self.time_since_update = 0
        
        # Daha hızlı onaylama: İlk tespitte hemen track_id verelim
        if self.state == 'new' and self.hits >= 1:  # Sadece bir kez görünmesi yeterli
            self.state = 'tracked'
            self.track_id = STrack._next_id
            STrack._next_id += 1
        elif self.state == 'lost':
            self.state = 'tracked'
            
        # Add current position to history
        self.history.append(self.tlbr)
        
    def _bbox_to_measurement(self, bbox):
        """Convert bbox [x1, y1, x2, y2] to measurement [cx, cy, aspect_ratio, height]."""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        aspect_ratio = w / h
        
        return np.array([cx, cy, aspect_ratio, h], dtype=np.float32)
    
    @property
    def tlbr(self):
        """Convert Kalman state to tlbr bbox format."""
        x, y, aspect_ratio, h = self.mean[:4]
        w = aspect_ratio * h
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return np.array([x1, y1, x2, y2], dtype=np.float32)
    
    def mark_lost(self):
        """Mark this track as lost."""
        self.state = 'lost'
        
    def is_confirmed(self):
        """Return if this track is confirmed."""
        # Sadece 'tracked' durumunda olması yeterli, ID kontrolüne gerek yok
        return self.state == 'tracked'
    
    def is_lost(self):
        """Return if this track is lost."""
        return self.state == 'lost'
    
    @staticmethod
    def reset_id():
        """Reset class ID counter."""
        STrack._next_id = 1


class Detection:
    """Detection class for ByteTrack."""
    def __init__(self, bbox, confidence, feature=None):
        self.bbox = np.array(bbox, dtype=np.float32)  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.feature = feature  # Optional appearance feature


class ByteTrack:
    """
    ByteTrack implementation for multi-object tracking.
    
    ByteTrack: https://arxiv.org/abs/2110.06864
    
    The main idea is to associate high-confidence detections first,
    then associate the unmatched trackers with the low-confidence detections.
    """
    def __init__(self, high_threshold=0.3, low_threshold=0.01, max_time_lost=60):
        self.tracked_tracks = []  # Confirmed tracks being tracked
        self.lost_tracks = []     # Lost tracks
        self.removed_tracks = []  # Removed tracks
        
        self.frame_count = 0
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.max_time_lost = max_time_lost  # Maximum frames to keep lost tracks
        
    def update(self, detection_results):
        """
        Update tracks using detection results from the current frame.
        
        Args:
            detection_results: List of Detection objects
            
        Returns:
            List of confirmed STrack objects
        """
        self.frame_count += 1
        
        # İlk aşamada tüm tespitleri takip et
        if not self.tracked_tracks and len(detection_results) > 0:
            # İlk tespit: Tüm tespitleri takip etmeye başla
            for det in detection_results:
                track = STrack(det.bbox, det.confidence, det.feature)
                # Hemen takip etmeye başla
                track.state = 'tracked'
                track.track_id = STrack._next_id
                STrack._next_id += 1
                self.tracked_tracks.append(track)
            
            # İlk durumda hemen sonuç döndür
            return self.tracked_tracks
        
        # Get high and low confidence detections
        high_dets = [d for d in detection_results if d.confidence >= self.high_threshold]
        low_dets = [d for d in detection_results if self.low_threshold <= d.confidence < self.high_threshold]
        
        # Get lists for updating
        tracked_tracks = []
        unconfirmed_tracks = []
        
        # Separate tracks into confirmed and unconfirmed
        for track in self.tracked_tracks:
            if track.is_confirmed():
                tracked_tracks.append(track)
            else:
                unconfirmed_tracks.append(track)
                
        # Step 1: Predict new locations for all tracks
        # And join tracked_stracks and lost_tracks
        all_tracks = tracked_tracks + self.lost_tracks
        for track in all_tracks:
            track.predict()
            
        # Step 2: Match high confidence detections with tracked tracks
        matched_track_indices, matched_det_indices, unmatched_track_indices, unmatched_det_indices = \
            self._match_tracks_with_detections(tracked_tracks, high_dets)
        
        # Update matched tracks with assigned detections
        for track_idx, det_idx in zip(matched_track_indices, matched_det_indices):
            tracked_tracks[track_idx].update(high_dets[det_idx])
            
        # Step 3: Match remaining tracks with low confidence detections
        # Get unmatched tracks
        unmatched_tracks = [tracked_tracks[i] for i in unmatched_track_indices]
        
        # Match unmatched tracks with low confidence detections
        if low_dets and unmatched_tracks:
            matched_track_indices2, matched_det_indices2, unmatched_track_indices2, _ = \
                self._match_tracks_with_detections(unmatched_tracks, low_dets)
                
            # Update matched tracks with low confidence detections
            for track_idx, det_idx in zip(matched_track_indices2, matched_det_indices2):
                unmatched_tracks[track_idx].update(low_dets[det_idx])
                
            # Get final unmatched tracks
            unmatched_tracks = [unmatched_tracks[i] for i in unmatched_track_indices2]
            
        # Deal with unmatched tracks
        for track in unmatched_tracks:
            if track.time_since_update >= self.max_time_lost:
                track.mark_lost()
                self.lost_tracks.append(track)
            else:
                self.tracked_tracks.append(track)
                
        # Deal with unmatched high confidence detections - create new tracks
        for idx in unmatched_det_indices:
            new_track = STrack(high_dets[idx].bbox, high_dets[idx].confidence, high_dets[idx].feature)
            # Hemen takip etmeye başla
            new_track.state = 'tracked'
            new_track.track_id = STrack._next_id
            STrack._next_id += 1
            self.tracked_tracks.append(new_track)
            
        # Update lists
        self.tracked_tracks = [t for t in self.tracked_tracks if not t.is_lost()]
        
        # Deal with lost tracks
        self.lost_tracks = [t for t in self.lost_tracks if t.time_since_update < self.max_time_lost]
        
        # Return all tracks (hem onaylanmış hem onaylanmamış)
        return self.tracked_tracks
        
    def _match_tracks_with_detections(self, tracks, detections):
        """
        Match tracks with detections.
        
        Args:
            tracks: List of STrack objects
            detections: List of Detection objects
            
        Returns:
            matched_track_indices, matched_det_indices, unmatched_track_indices, unmatched_det_indices
        """
        if not tracks or not detections:
            return [], [], list(range(len(tracks))), list(range(len(detections)))
            
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._calculate_iou(track.tlbr, det.bbox)
                
        # Apply Hungarian algorithm for matching
        if min(iou_matrix.shape) > 0:
            # Use linear_sum_assignment to solve the assignment problem
            row_indices, col_indices = linear_sum_assignment(-iou_matrix)
            matched_indices = np.column_stack((row_indices, col_indices))
        else:
            matched_indices = np.empty((0, 2), dtype=int)
            
        # Filter matches using threshold
        matches = []
        unmatched_track_indices = []
        unmatched_det_indices = list(range(len(detections)))
        
        for i, j in matched_indices:
            if iou_matrix[i, j] < 0.25:  # Lower IoU threshold for matching
                unmatched_track_indices.append(i)
            else:
                matches.append((i, j))
                if j in unmatched_det_indices:
                    unmatched_det_indices.remove(j)
                    
        # Get corresponding indices
        matched_track_indices = [match[0] for match in matches]
        matched_det_indices = [match[1] for match in matches]
        
        # Add remaining unmatched track indices
        for i in range(len(tracks)):
            if i not in matched_track_indices and i not in unmatched_track_indices:
                unmatched_track_indices.append(i)
                
        return matched_track_indices, matched_det_indices, unmatched_track_indices, unmatched_det_indices
    
    @staticmethod
    def _calculate_iou(box1, box2):
        """
        Calculate IoU between two boxes.
        
        Args:
            box1, box2: [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        # Get intersection coordinates
        xx1 = max(box1[0], box2[0])
        yy1 = max(box1[1], box2[1])
        xx2 = min(box1[2], box2[2])
        yy2 = min(box1[3], box2[3])
        
        # Calculate areas
        w = max(0, xx2 - xx1)
        h = max(0, yy2 - yy1)
        intersection = w * h
        
        # Calculate union
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        # Return IoU
        return intersection / max(union, 1e-6) 