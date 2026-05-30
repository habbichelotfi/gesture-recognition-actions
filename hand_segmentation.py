"""
Hand segmentation and finger counting module.
Combines and modernizes functionality from recogize.py and segment.py.
"""

import logging
import numpy as np
from typing import Tuple, Optional

import cv2
from sklearn.metrics import pairwise
import imutils

import config
from utils import get_contours, get_largest_contour

logger = logging.getLogger(__name__)


class HandSegmenter:
    """Handles hand segmentation and finger counting."""

    def __init__(
        self,
        accumulation_weight: float = config.ACCUMULATION_WEIGHT,
        segmentation_threshold: int = config.SEGMENTATION_THRESHOLD
    ):
        """
        Initialize hand segmenter.

        Args:
            accumulation_weight: Weight for background accumulation
            segmentation_threshold: Threshold for background subtraction
        """
        self.accumulation_weight = accumulation_weight
        self.segmentation_threshold = segmentation_threshold
        self.background = None
        self.calibrated = False
        self.calibration_frames = 30

        logger.info("HandSegmenter initialized")

    def update_background(self, image: np.ndarray):
        """
        Update background model using running average.

        Args:
            image: Current frame (grayscale)
        """
        if self.background is None:
            self.background = image.copy().astype("float")
            return

        cv2.accumulateWeighted(
            image,
            self.background,
            self.accumulation_weight
        )

    def segment_hand(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Segment hand region from image.

        Args:
            image: Input grayscale image

        Returns:
            Tuple of (thresholded_image, hand_contour) or None
        """
        if self.background is None:
            logger.warning("Background not initialized")
            return None

        # Calculate difference from background
        diff = cv2.absdiff(self.background.astype("uint8"), image)

        # Threshold to get foreground
        _, thresholded = cv2.threshold(
            diff,
            self.segmentation_threshold,
            255,
            cv2.THRESH_BINARY
        )

        # Get contours
        contours = get_contours(thresholded)

        if not contours:
            return None

        # Get largest contour (hand)
        hand_contour = get_largest_contour(contours)

        return thresholded, hand_contour

    def calibrate(self, frames: int = None):
        """Mark segmenter as calibrated."""
        self.calibrated = True
        logger.info("Hand segmenter calibrated")

    def count_fingers(
        self,
        thresholded: np.ndarray,
        hand_contour: np.ndarray
    ) -> int:
        """
        Count fingers in hand region.

        Args:
            thresholded: Binary thresholded image
            hand_contour: Hand contour

        Returns:
            Number of fingers detected
        """
        # Find convex hull
        hull = cv2.convexHull(hand_contour)

        # Find extreme points
        extreme_top = tuple(hull[hull[:, :, 1].argmin()][0])
        extreme_bottom = tuple(hull[hull[:, :, 1].argmax()][0])
        extreme_left = tuple(hull[hull[:, :, 0].argmin()][0])
        extreme_right = tuple(hull[hull[:, :, 0].argmax()][0])

        # Calculate palm center
        palm_center_x = int((extreme_left[0] + extreme_right[0]) / 2)
        palm_center_y = int((extreme_top[1] + extreme_bottom[1]) / 2)

        # Calculate radius based on max distance
        distances = pairwise.euclidean_distances(
            [(palm_center_x, palm_center_y)],
            Y=[extreme_left, extreme_right, extreme_top, extreme_bottom]
        )[0]

        max_distance = distances[distances.argmax()]
        radius = int(config.FINGER_RADIUS_MULTIPLIER * max_distance)

        # Create circular ROI
        circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
        cv2.circle(circular_roi, (palm_center_x, palm_center_y), radius, 255, 1)

        # Apply mask
        masked_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

        # Find contours in ROI
        contours = get_contours(masked_roi)

        # Calculate circumference
        circumference = 2 * np.pi * radius

        # Count fingers
        finger_count = 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Check if contour is a finger (not wrist)
            is_not_wrist = (palm_center_y + (palm_center_y * config.FINGER_WRIST_MULTIPLIER)) > (y + h)
            is_finger_size = (circumference * config.FINGER_CIRCUMFERENCE_MULTIPLIER) > contour.shape[0]

            if is_not_wrist and is_finger_size:
                finger_count += 1

        return finger_count

    def draw_hand(
        self,
        image: np.ndarray,
        hand_contour: np.ndarray,
        roi_coords: Tuple[int, int, int, int],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw hand contour on image.

        Args:
            image: Input image
            hand_contour: Hand contour
            roi_coords: ROI coordinates (top, right, bottom, left)
            color: Line color (BGR)
            thickness: Line thickness

        Returns:
            Image with drawn contour
        """
        top, right, bottom, left = roi_coords
        offset = (right, top)

        cv2.drawContours(
            image,
            [hand_contour + np.array(offset)],
            -1,
            color,
            thickness
        )

        return image


class ROI:
    """Region of Interest manager."""

    def __init__(
        self,
        top: int = config.ROI_TOP,
        right: int = config.ROI_RIGHT,
        bottom: int = config.ROI_BOTTOM,
        left: int = config.ROI_LEFT
    ):
        """
        Initialize ROI.

        Args:
            top, right, bottom, left: ROI boundaries
        """
        self.top = top
        self.right = right
        self.bottom = bottom
        self.left = left

    def extract(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract ROI from frame.

        Args:
            frame: Input frame

        Returns:
            ROI region
        """
        return frame[self.top:self.bottom, self.left:self.right]

    def get_coordinates(self) -> Tuple[int, int, int, int]:
        """Get ROI coordinates."""
        return (self.top, self.right, self.bottom, self.left)

    def draw_rectangle(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw ROI rectangle on frame.

        Args:
            frame: Input frame
            color: Rectangle color
            thickness: Line thickness

        Returns:
            Frame with drawn rectangle
        """
        cv2.rectangle(
            frame,
            (self.left, self.top),
            (self.right, self.bottom),
            color,
            thickness
        )

        return frame


def process_frame_for_segmentation(
    frame: np.ndarray,
    roi: ROI
) -> np.ndarray:
    """
    Process frame for hand segmentation.

    Args:
        frame: Input frame
        roi: Region of interest

    Returns:
        Processed frame (grayscale, blurred)
    """
    # Extract ROI
    roi_frame = roi.extract(frame)

    # Convert to grayscale
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, config.BLUR_KERNEL_SIZE, 0)

    return blurred


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Hand segmentation module loaded successfully")

