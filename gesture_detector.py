"""
Gesture detection module using dlib SVM models.
Detects hand gestures and triggers corresponding actions.
"""

import logging
import time
from typing import List, Tuple, Optional

import numpy as np
import cv2
import dlib
import pyautogui

import config
from utils import validate_file_exists

logger = logging.getLogger(__name__)


class SVMGestureDetector:
    """Detects gestures using dlib SVM models."""

    def __init__(self, model_confidence_threshold: float = config.SVM_CONFIDENCE_THRESHOLD):
        """
        Initialize gesture detector.

        Args:
            model_confidence_threshold: Minimum confidence for detection
        """
        self.detectors = []
        self.gesture_names = config.SVM_GESTURE_NAMES
        self.confidence_threshold = model_confidence_threshold
        self.last_action_time = {}
        self.cooldown_ms = config.COOLDOWN_MS

        self._load_models()
        logger.info("SVMGestureDetector initialized")

    def _load_models(self):
        """Load SVM models from files."""
        for gesture_name, model_path in config.SVM_MODELS.items():
            if not validate_file_exists(model_path):
                logger.warning(f"Model not found: {model_path}")
                continue

            try:
                detector = dlib.fhog_object_detector(model_path)
                self.detectors.append(detector)
                logger.info(f"Loaded model: {gesture_name} from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model {gesture_name}: {str(e)}")

    def detect_gestures(
        self,
        frame: np.ndarray,
        scale_factor: int = 1
    ) -> List[Tuple[dict, int, float]]:
        """
        Detect gestures in frame.

        Args:
            frame: Input frame
            scale_factor: Downsampling factor for faster processing

        Returns:
            List of detections with format (detection_dict, gesture_idx, confidence)
        """
        if not self.detectors:
            logger.warning("No detectors available")
            return []

        # Downsample frame for faster processing
        if scale_factor > 1:
            new_width = int(frame.shape[1] / scale_factor)
            new_height = int(frame.shape[0] / scale_factor)
            frame_downsampled = cv2.resize(frame, (new_width, new_height))
        else:
            frame_downsampled = frame

        try:
            # Run all detectors
            detections, confidences, detector_idxs = dlib.fhog_object_detector.run_multiple(
                self.detectors,
                frame_downsampled,
                upsample_num_times=config.SVM_UPSAMPLE_NUM_TIMES,
                adjust_threshold=config.SVM_ADJUST_THRESHOLD
            )
        except Exception as e:
            logger.error(f"Error during gesture detection: {str(e)}")
            return []

        # Process detections
        results = []

        for i, (detection, confidence, detector_idx) in enumerate(
            zip(detections, confidences, detector_idxs)
        ):
            # Scale back to original frame size
            if scale_factor > 1:
                x1 = int(detection.left() * scale_factor)
                y1 = int(detection.top() * scale_factor)
                x2 = int(detection.right() * scale_factor)
                y2 = int(detection.bottom() * scale_factor)
            else:
                x1 = int(detection.left())
                y1 = int(detection.top())
                x2 = int(detection.right())
                y2 = int(detection.bottom())

            detection_dict = {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": x2 - x1,
                "height": y2 - y1,
                "area": (x2 - x1) * (y2 - y1),
                "center_x": (x1 + x2) // 2,
                "center_y": (y1 + y2) // 2
            }

            results.append((detection_dict, detector_idx, confidence))

        return results

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Tuple[dict, int, float]],
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        Draw detection boxes and labels on frame.

        Args:
            frame: Input frame
            detections: List of detections
            show_confidence: Whether to show confidence scores

        Returns:
            Frame with drawn detections
        """
        for detection_dict, gesture_idx, confidence in detections:
            if confidence < self.confidence_threshold:
                continue

            x1, y1 = detection_dict["x1"], detection_dict["y1"]
            x2, y2 = detection_dict["x2"], detection_dict["y2"]

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            gesture_name = self.gesture_names[gesture_idx]
            confidence_pct = confidence * 100

            if show_confidence:
                label = f"{gesture_name}: {confidence_pct:.1f}%"
            else:
                label = gesture_name

            cv2.putText(
                frame,
                label,
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_COMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        return frame

    def can_perform_action(self, gesture_name: str) -> bool:
        """
        Check if enough time has passed since last action.

        Args:
            gesture_name: Name of gesture

        Returns:
            True if action can be performed
        """
        current_time = time.time() * 1000  # Convert to milliseconds
        last_time = self.last_action_time.get(gesture_name, 0)

        if current_time - last_time > self.cooldown_ms:
            self.last_action_time[gesture_name] = current_time
            return True

        return False

    def perform_action(self, gesture_name: str):
        """
        Perform action associated with gesture.

        Args:
            gesture_name: Name of gesture to perform action for
        """
        if not self.can_perform_action(gesture_name):
            return

        action = config.GESTURE_ACTIONS.get(gesture_name)

        if action is None:
            logger.warning(f"No action configured for gesture: {gesture_name}")
            return

        action_type = action[0]
        action_params = action[1:]

        try:
            if action_type == "press":
                pyautogui.press(action_params[0])
                logger.info(f"Performed action: press({action_params[0]})")

            elif action_type == "scroll":
                pyautogui.scroll(action_params[0])
                logger.info(f"Performed action: scroll({action_params[0]})")

            elif action_type == "hotkey":
                pyautogui.hotkey(*action_params[0])
                logger.info(f"Performed action: hotkey{action_params[0]}")

            else:
                logger.warning(f"Unknown action type: {action_type}")

        except Exception as e:
            logger.error(f"Error performing action {gesture_name}: {str(e)}")


def process_detections_and_perform_actions(
    detector: SVMGestureDetector,
    detections: List[Tuple[dict, int, float]]
):
    """
    Process detections and perform actions for high-confidence detections.

    Args:
        detector: Gesture detector instance
        detections: List of detections
    """
    for detection_dict, gesture_idx, confidence in detections:
        if confidence < detector.confidence_threshold:
            continue

        gesture_name = detector.gesture_names[gesture_idx]
        detector.perform_action(gesture_name)


if __name__ == "__main__":
    import numpy as np

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Gesture detection module loaded successfully")

