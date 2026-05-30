"""
Utility functions and helper classes for gesture recognition.
"""

import cv2
import logging
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from config import LOG_FORMAT, LOG_LEVEL, PROJECT_ROOT

logger = logging.getLogger(__name__)


def setup_logging(log_file: str = None) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_file: Optional path to log file

    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file or str(PROJECT_ROOT / "gesture_recognition.log"))
        ]
    )
    return logging.getLogger(__name__)


def load_image(image_path: str, grayscale: bool = False) -> Optional[np.ndarray]:
    """
    Load image from file with error handling.

    Args:
        image_path: Path to image file
        grayscale: Whether to convert to grayscale

    Returns:
        Image array or None if loading fails
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return None

        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        return None


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image to specified size.

    Args:
        image: Input image array
        size: Target size (width, height)

    Returns:
        Resized image
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to 0-1 range.

    Args:
        image: Input image

    Returns:
        Normalized image
    """
    return image.astype("float32") / 255.0


def apply_gaussian_blur(image: np.ndarray, kernel_size: Tuple[int, int] = (7, 7)) -> np.ndarray:
    """
    Apply Gaussian blur to image.

    Args:
        image: Input image
        kernel_size: Size of the kernel

    Returns:
        Blurred image
    """
    return cv2.GaussianBlur(image, kernel_size, 0)


def get_contours(image: np.ndarray) -> list:
    """
    Get contours from binary image. Handles OpenCV version differences.

    Args:
        image: Binary image

    Returns:
        List of contours
    """
    try:
        contours, _ = cv2.findContours(
            image.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return contours
    except ValueError:
        # For older OpenCV versions that return 3 values
        _, contours, _ = cv2.findContours(
            image.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return contours


def get_largest_contour(contours: list) -> Optional[np.ndarray]:
    """
    Find the largest contour by area.

    Args:
        contours: List of contours

    Returns:
        Largest contour or None
    """
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


class FpsCounter:
    """Utility class for calculating FPS."""

    def __init__(self, update_interval: int = 30):
        """
        Initialize FPS counter.

        Args:
            update_interval: Number of frames between updates
        """
        self.start_time = None
        self.frame_count = 0
        self.fps = 0
        self.update_interval = update_interval
        self.reset()

    def reset(self):
        """Reset counter."""
        import time
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0

    def update(self) -> float:
        """
        Update counter and return current FPS.

        Returns:
            Current FPS value
        """
        import time
        self.frame_count += 1

        if self.frame_count % self.update_interval == 0:
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed if elapsed > 0 else 0

        return self.fps

    def get_fps(self) -> float:
        """Get current FPS value."""
        return self.fps


class CameraManager:
    """Manages webcam initialization and error handling."""

    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        """
        Initialize camera manager.

        Args:
            camera_id: Camera device ID
            width: Frame width
            height: Frame height
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        self.is_opened = False

    def open(self) -> bool:
        """
        Open camera.

        Returns:
            True if successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_id)

            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return False

            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency

            self.is_opened = True
            logger.info(f"Camera {self.camera_id} opened successfully")
            return True
        except Exception as e:
            logger.error(f"Error opening camera: {str(e)}")
            return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read frame from camera.

        Returns:
            Tuple of (success, frame)
        """
        if not self.is_opened:
            logger.error("Camera is not opened")
            return False, None

        try:
            success, frame = self.cap.read()
            return success, frame
        except Exception as e:
            logger.error(f"Error reading frame: {str(e)}")
            return False, None

    def release(self):
        """Release camera resources."""
        if self.cap:
            self.cap.release()
            self.is_opened = False
            logger.info("Camera released")

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


def validate_file_exists(file_path: str) -> bool:
    """
    Validate that file exists.

    Args:
        file_path: Path to file

    Returns:
        True if file exists, False otherwise
    """
    if Path(file_path).exists():
        return True

    logger.error(f"File not found: {file_path}")
    return False


def get_detection_label(gesture_names: list, index: int) -> str:
    """
    Safely get gesture label by index.

    Args:
        gesture_names: List of gesture names
        index: Index of gesture

    Returns:
        Gesture name or "Unknown"
    """
    if 0 <= index < len(gesture_names):
        return gesture_names[index]

    logger.warning(f"Invalid gesture index: {index}")
    return "Unknown"

