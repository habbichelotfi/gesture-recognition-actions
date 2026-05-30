"""
Main gesture recognition application.
Real-time hand gesture recognition and action automation.
"""

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

import config
from utils import setup_logging, CameraManager, FpsCounter, validate_file_exists
from gesture_detector import SVMGestureDetector, process_detections_and_perform_actions
from hand_segmentation import HandSegmenter, ROI, process_frame_for_segmentation

# Setup logging
logger = setup_logging()


class GestureRecognitionApp:
    """Main gesture recognition application."""

    def __init__(self, use_svm: bool = True):
        """
        Initialize application.

        Args:
            use_svm: Use SVM-based detection if True, otherwise CNN-based
        """
        self.use_svm = use_svm
        self.running = False

        # Initialize components
        self.camera_manager = CameraManager(
            camera_id=config.CAMERA_ID,
            width=config.CAMERA_WIDTH,
            height=config.CAMERA_HEIGHT
        )
        self.fps_counter = FpsCounter(update_interval=config.FPS_UPDATE_INTERVAL)

        if use_svm:
            self.detector = SVMGestureDetector()
            self.hand_segmenter = HandSegmenter()
            self.roi = ROI()

        logger.info(f"GestureRecognitionApp initialized (SVM: {use_svm})")

    def setup(self) -> bool:
        """
        Setup application resources.

        Returns:
            True if successful
        """
        logger.info("Setting up application...")

        # Open camera
        if not self.camera_manager.open():
            logger.error("Failed to open camera")
            return False

        logger.info("Application setup completed")
        return True

    def cleanup(self):
        """Cleanup application resources."""
        logger.info("Cleaning up application...")
        self.camera_manager.release()
        cv2.destroyAllWindows()
        logger.info("Cleanup completed")

    def run_svm_mode(self):
        """Run gesture recognition in SVM mode."""
        logger.info("Starting SVM gesture recognition...")

        self.running = True
        frame_count = 0

        try:
            while self.running:
                # Read frame
                success, frame = self.camera_manager.read_frame()

                if not success:
                    logger.warning("Failed to read frame")
                    break

                # Flip frame
                frame = cv2.flip(frame, 1)

                # Update FPS
                fps = self.fps_counter.update()

                # Detect gestures
                detections = self.detector.detect_gestures(
                    frame,
                    scale_factor=config.SCALE_FACTOR
                )

                # Draw detections
                if config.DISPLAY_CONFIDENCE:
                    frame = self.detector.draw_detections(frame, detections)

                # Perform actions
                process_detections_and_perform_actions(self.detector, detections)

                # Display FPS
                if config.DISPLAY_FPS:
                    cv2.putText(
                        frame,
                        f"FPS: {fps:.2f}",
                        (20, 20),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

                # Display number of detections
                cv2.putText(
                    frame,
                    f"Detections: {len(detections)}",
                    (20, 50),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

                # Show frame
                cv2.imshow(config.DISPLAY_WINDOW_NAME, frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit requested by user")
                    break

                frame_count += 1

                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}", exc_info=True)
        finally:
            self.running = False

    def run_segmentation_mode(self):
        """Run hand segmentation mode (finger counting)."""
        logger.info("Starting hand segmentation mode...")

        self.running = True
        frame_count = 0
        calibration_progress = 0

        try:
            while self.running:
                # Read frame
                success, frame = self.camera_manager.read_frame()

                if not success:
                    logger.warning("Failed to read frame")
                    break

                # Flip frame
                frame = cv2.flip(frame, 1)
                frame_copy = frame.copy()

                # Update FPS
                fps = self.fps_counter.update()

                # Extract and process ROI
                roi_frame = process_frame_for_segmentation(frame, self.roi)

                # Calibration phase
                if not self.hand_segmenter.calibrated:
                    self.hand_segmenter.update_background(roi_frame)
                    calibration_progress += 1

                    if calibration_progress >= self.hand_segmenter.calibration_frames:
                        self.hand_segmenter.calibrate()
                        logger.info("Calibration complete")

                    status = f"Calibrating... {calibration_progress}/{self.hand_segmenter.calibration_frames}"
                    cv2.putText(
                        frame_copy,
                        status,
                        (20, 40),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

                else:
                    # Segment hand
                    segmentation_result = self.hand_segmenter.segment_hand(roi_frame)

                    if segmentation_result is not None:
                        thresholded, hand_contour = segmentation_result

                        # Count fingers
                        finger_count = self.hand_segmenter.count_fingers(thresholded, hand_contour)

                        # Draw hand contour
                        self.hand_segmenter.draw_hand(
                            frame_copy,
                            hand_contour,
                            self.roi.get_coordinates()
                        )

                        # Display finger count
                        cv2.putText(
                            frame_copy,
                            f"Fingers: {finger_count}",
                            (20, 40),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.8,
                            (0, 0, 255),
                            2
                        )

                        # Display thresholded image
                        cv2.imshow("Thresholded", thresholded)

                # Draw ROI rectangle
                self.roi.draw_rectangle(frame_copy)

                # Display FPS
                if config.DISPLAY_FPS:
                    cv2.putText(
                        frame_copy,
                        f"FPS: {fps:.2f}",
                        (20, 20),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.6,
                        (0, 0, 255),
                        2
                    )

                # Show frame
                cv2.imshow(config.DISPLAY_WINDOW_NAME, frame_copy)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit requested by user")
                    break

                frame_count += 1

                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in segmentation mode: {str(e)}", exc_info=True)
        finally:
            self.running = False

    def run(self, mode: str = "svm"):
        """
        Run application.

        Args:
            mode: "svm" for gesture detection, "segment" for finger counting
        """
        if mode == "svm":
            self.run_svm_mode()
        elif mode == "segment":
            self.run_segmentation_mode()
        else:
            logger.error(f"Unknown mode: {mode}")

    def __enter__(self):
        """Context manager entry."""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


def main():
    """Main entry point."""
    logger.info("=" * 80)
    logger.info("Gesture Recognition Application")
    logger.info("=" * 80)

    import argparse

    parser = argparse.ArgumentParser(description="Gesture Recognition System")
    parser.add_argument(
        "--mode",
        choices=["svm", "segment"],
        default="svm",
        help="Operating mode (svm=gesture detection, segment=finger counting)"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=config.CAMERA_ID,
        help="Camera device ID"
    )

    args = parser.parse_args()

    # Override camera ID if specified
    if args.camera != config.CAMERA_ID:
        config.CAMERA_ID = args.camera
        logger.info(f"Using camera ID: {args.camera}")

    # Run application
    try:
        with GestureRecognitionApp(use_svm=True) as app:
            if not app.setup():
                logger.error("Failed to setup application")
                return 1

            app.run(mode=args.mode)

        logger.info("Application exited successfully")
        return 0

    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

