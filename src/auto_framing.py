#!/usr/bin/env python3
"""
Auto-Framing Module - Subject Detection and Composition

Detects people in the frame and calculates optimal gimbal angles
to keep them properly framed for photos/video.

Features:
    - Face detection using Haar cascades or DNN
    - Full-body detection using HOG or YOLO
    - Rule of thirds composition
    - Headroom calculation
    - Multi-subject framing
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class Subject:
    """Detected subject information."""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    center: Tuple[int, int]          # x, y
    confidence: float
    subject_type: str  # "face", "body", "person"
    
    @property
    def area(self) -> int:
        return self.bbox[2] * self.bbox[3]


@dataclass
class FramingData:
    """Output of framing calculation."""
    detected: bool
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    confidence: float
    optimal_pitch: float = 90.0
    optimal_yaw: float = 90.0


class AutoFramer:
    """
    Auto-framing system for camera gimbal.
    
    Detects subjects and calculates servo angles to maintain
    optimal composition.
    """
    
    def __init__(
        self,
        smoothing: float = 0.15,
        model_type: str = "yolo",
        composition: str = "center",  # "center", "rule_of_thirds"
        headroom_ratio: float = 0.15  # Space above head
    ):
        self.smoothing = smoothing
        self.model_type = model_type
        self.composition = composition
        self.headroom_ratio = headroom_ratio
        
        # Detection models
        self._face_cascade = None
        self._body_detector = None
        self._dnn_net = None
        
        # Tracking state
        self._last_subject: Optional[Subject] = None
        self._lost_frames = 0
        self._max_lost_frames = 30
        
        # Smoothing buffers
        self._center_buffer = deque(maxlen=10)
        
        # Initialize models
        self._init_models()
        
        logger.info(f"AutoFramer initialized with {model_type} detector")
    
    def _init_models(self):
        """Initialize detection models."""
        if self.model_type == "haar":
            self._face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            logger.info("Loaded Haar face cascade")
            
        elif self.model_type == "dnn":
            # OpenCV DNN face detector
            model_file = "models/res10_300x300_ssd_iter_140000.caffemodel"
            config_file = "models/deploy.prototxt"
            try:
                self._dnn_net = cv2.dnn.readNetFromCaffe(config_file, model_file)
                logger.info("Loaded DNN face detector")
            except Exception as e:
                logger.error(f"Failed to load DNN model: {e}")
                logger.info("Falling back to Haar cascade")
                self._face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                
        elif self.model_type == "yolo":
            # For now, fall back to Haar + HOG
            # Full YOLO implementation would require model files
            self._face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            self._body_detector = cv2.HOGDescriptor()
            self._body_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            logger.info("Loaded HOG people detector + Haar face")
    
    def process_frame(self, frame: np.ndarray) -> FramingData:
        """
        Process a frame and return framing data.
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            FramingData with detection results
        """
        subjects = self._detect_subjects(frame)
        
        if not subjects:
            self._lost_frames += 1
            if self._lost_frames > self._max_lost_frames:
                self._last_subject = None
                self._center_buffer.clear()
            
            return FramingData(
                detected=False,
                bbox=(0, 0, 0, 0),
                center=(frame.shape[1] // 2, frame.shape[0] // 2),
                confidence=0.0
            )
        
        # Select primary subject (largest for now, could use tracking)
        primary = max(subjects, key=lambda s: s.area)
        
        # Update tracking
        self._lost_frames = 0
        self._last_subject = primary
        self._center_buffer.append(primary.center)
        
        # Calculate smoothed center
        if len(self._center_buffer) > 0:
            avg_center = (
                int(np.mean([c[0] for c in self._center_buffer])),
                int(np.mean([c[1] for c in self._center_buffer]))
            )
        else:
            avg_center = primary.center
        
        return FramingData(
            detected=True,
            bbox=primary.bbox,
            center=avg_center,
            confidence=primary.confidence,
            optimal_pitch=90.0,
            optimal_yaw=90.0
        )
    
    def _detect_subjects(self, frame: np.ndarray) -> List[Subject]:
        """Detect all subjects in frame."""
        subjects = []
        
        if self.model_type == "haar":
            subjects = self._detect_faces_haar(frame)
        elif self.model_type == "dnn":
            subjects = self._detect_faces_dnn(frame)
        elif self.model_type == "yolo":
            # Combine face and body detection
            face_subjects = self._detect_faces_haar(frame)
            body_subjects = self._detect_bodies_hog(frame)
            subjects = self._merge_detections(face_subjects, body_subjects)
        
        return subjects
    
    def _detect_faces_haar(self, frame: np.ndarray) -> List[Subject]:
        """Detect faces using Haar cascades."""
        if self._face_cascade is None:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
            maxSize=(frame.shape[1] // 2, frame.shape[0] // 2)
        )
        
        subjects = []
        for x, y, w, h in faces:
            center = (x + w // 2, y + h // 2)
            # Estimate confidence based on detection quality
            confidence = min(1.0, (w * h) / (frame.shape[0] * frame.shape[1]) * 10)
            subjects.append(Subject(
                bbox=(x, y, w, h),
                center=center,
                confidence=confidence,
                subject_type="face"
            ))
        
        return subjects
    
    def _detect_faces_dnn(self, frame: np.ndarray) -> List[Subject]:
        """Detect faces using OpenCV DNN."""
        if self._dnn_net is None:
            return self._detect_faces_haar(frame)
        
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )
        
        self._dnn_net.setInput(blob)
        detections = self._dnn_net.forward()
        
        subjects = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                bw, bh = x2 - x1, y2 - y1
                center = (x1 + bw // 2, y1 + bh // 2)
                subjects.append(Subject(
                    bbox=(x1, y1, bw, bh),
                    center=center,
                    confidence=float(confidence),
                    subject_type="face"
                ))
        
        return subjects
    
    def _detect_bodies_hog(self, frame: np.ndarray) -> List[Subject]:
        """Detect full bodies using HOG."""
        if self._body_detector is None:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        bodies, weights = self._body_detector.detectMultiScale(
            gray,
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05
        )
        
        subjects = []
        for (x, y, w, h), weight in zip(bodies, weights):
            if weight > 0.5:  # Confidence threshold
                center = (x + w // 2, y + h // 2)
                subjects.append(Subject(
                    bbox=(x, y, w, h),
                    center=center,
                    confidence=float(weight),
                    subject_type="body"
                ))
        
        return subjects
    
    def _merge_detections(
        self,
        faces: List[Subject],
        bodies: List[Subject]
    ) -> List[Subject]:
        """Merge face and body detections."""
        # Simple approach: prefer faces, add bodies that don't overlap
        merged = list(faces)
        
        for body in bodies:
            overlaps = False
            for face in faces:
                # Check if face is inside body
                fx, fy, fw, fh = face.bbox
                bx, by, bw, bh = body.bbox
                
                if (bx < fx and fx + fw < bx + bw and
                    by < fy and fy + fh < by + bh):
                    overlaps = True
                    break
            
            if not overlaps:
                merged.append(body)
        
        return merged
    
    def calculate_framing(
        self,
        subject_bbox: Tuple[int, int, int, int],
        frame_size: Tuple[int, int]
    ) -> Tuple[float, float]:
        """
        Calculate optimal servo angles for framing.
        
        Args:
            subject_bbox: (x, y, w, h) of subject
            frame_size: (width, height) of frame
            
        Returns:
            Tuple of (pitch_angle, yaw_angle) in degrees
        """
        frame_w, frame_h = frame_size
        sx, sy, sw, sh = subject_bbox
        
        # Subject center
        subject_cx = sx + sw // 2
        subject_cy = sy + sh // 2
        
        # Frame center
        frame_cx = frame_w // 2
        frame_cy = frame_h // 2
        
        # Calculate target position based on composition
        if self.composition == "center":
            target_x = frame_cx
            target_y = frame_cy
        elif self.composition == "rule_of_thirds":
            # Place subject at left or right third intersection
            # For now, use center with slight headroom adjustment
            target_x = frame_cx
            # Place eyes at upper third line (approximately)
            target_y = int(frame_h * 0.33)
        else:
            target_x = frame_cx
            target_y = frame_cy
        
        # Apply headroom (move target up by headroom ratio)
        target_y -= int(sh * self.headroom_ratio)
        
        # Calculate pixel error
        error_x = subject_cx - target_x
        error_y = subject_cy - target_y
        
        # Convert to angles (assuming camera FOV)
        fov_h = 60.0  # Horizontal FOV in degrees
        fov_v = 45.0  # Vertical FOV in degrees
        
        # Scale: degrees per pixel
        scale_x = fov_h / frame_w
        scale_y = fov_v / frame_h
        
        # Calculate angle adjustments
        yaw_adjustment = error_x * scale_x
        pitch_adjustment = error_y * scale_y
        
        # Current gimbal position (assumed centered at 90)
        current_yaw = 90.0
        current_pitch = 90.0
        
        # Calculate new target angles
        target_yaw = current_yaw + yaw_adjustment
        target_pitch = current_pitch - pitch_adjustment  # Inverted: up is negative
        
        # Clamp to servo limits
        target_yaw = max(0, min(180, target_yaw))
        target_pitch = max(0, min(180, target_pitch))
        
        return (target_pitch, target_yaw)
    
    def calculate_multi_subject_framing(
        self,
        subjects: List[Subject],
        frame_size: Tuple[int, int]
    ) -> Tuple[float, float]:
        """
        Calculate framing to include multiple subjects.
        
        Finds the bounding box that contains all subjects and
        frames to include them all with padding.
        """
        if not subjects:
            return (90.0, 90.0)
        
        # Find bounding box of all subjects
        min_x = min(s.bbox[0] for s in subjects)
        min_y = min(s.bbox[1] for s in subjects)
        max_x = max(s.bbox[0] + s.bbox[2] for s in subjects)
        max_y = max(s.bbox[1] + s.bbox[3] for s in subjects)
        
        # Add padding
        padding = 50
        min_x -= padding
        min_y -= padding
        max_x += padding
        max_y += padding
        
        # Calculate center
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        
        # Convert to angles
        frame_w, frame_h = frame_size
        error_x = center_x - frame_w // 2
        error_y = center_y - frame_h // 2
        
        yaw = 90.0 + (error_x / frame_w) * 60.0
        pitch = 90.0 - (error_y / frame_h) * 45.0
        
        return (
            max(0, min(180, pitch)),
            max(0, min(180, yaw))
        )
