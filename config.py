"""
Configuration settings for Gesture Recognition system.
Centralizes all hardcoded values for easy management and customization.
"""

from pathlib import Path
from typing import List, Dict
import logging

# ==================== Paths ====================
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
DATASETS_DIR = PROJECT_ROOT / "datasets"
BOXES_DIR = PROJECT_ROOT / "boxes"
RESULTS_DIR = PROJECT_ROOT / "result"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
DATASETS_DIR.mkdir(exist_ok=True)
BOXES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ==================== Camera Settings ====================
CAMERA_ID: int = 0  # Primary camera (change to 1 for secondary)
CAMERA_WIDTH: int = 1280
CAMERA_HEIGHT: int = 720
CAMERA_FPS: int = 30

# ==================== Gesture Recognition Models ====================
# SVM Models for gesture detection
SVM_MODELS = {
    "pause": str(MODELS_DIR / "Pause_detector.svm"),
    "scrolling": str(MODELS_DIR / "Scrolling_up.svm"),
    "scrolling_tabs": str(MODELS_DIR / "Scrolling_tabs.svm"),
    "change_program": str(MODELS_DIR / "change_programe.svm"),
}

SVM_GESTURE_NAMES: List[str] = [
    "Pause", "Scrolling Up", "Scrolling Tabs", "Change Program"
]

SVM_CONFIDENCE_THRESHOLD: float = 0.90  # 90% confidence minimum
SVM_UPSAMPLE_NUM_TIMES: int = 1
SVM_ADJUST_THRESHOLD: float = 0.0

# ==================== CNN Model Settings ====================
CNN_MODEL_PATH = str(MODELS_DIR / "gesture_cnn_model.h5")
CNN_GESTURE_NAMES: List[str] = [
    "A", "C", "D", "E", "F", "G", "H", "K", "L", "N", "P", "Q", "S", "U", "V", "W", "Z"
]
CNN_NUM_CLASSES = len(CNN_GESTURE_NAMES)  # 17 gestures

# CNN Training parameters
CNN_IMAGE_SIZE: tuple = (300, 300)
CNN_BATCH_SIZE: int = 32
CNN_EPOCHS: int = 20
CNN_VALIDATION_SPLIT: float = 0.3
CNN_RANDOM_STATE: int = 42

# ==================== Hand Segmentation Settings ====================
ROI_TOP: int = 10
ROI_RIGHT: int = 350
ROI_BOTTOM: int = 225
ROI_LEFT: int = 590

# Background subtraction
ACCUMULATION_WEIGHT: float = 0.5
SEGMENTATION_THRESHOLD: int = 25

# Gaussian blur
BLUR_KERNEL_SIZE: tuple = (7, 7)

# Finger counting
FINGER_RADIUS_MULTIPLIER: float = 0.8
FINGER_CIRCUMFERENCE_MULTIPLIER: float = 0.25
FINGER_WRIST_MULTIPLIER: float = 0.25

# ==================== Training Data Collection ====================
TRAINING_WINDOW_WIDTH: int = 190
TRAINING_WINDOW_HEIGHT: int = 190
TRAINING_SKIP_FRAMES: int = 3
TRAINING_INITIAL_WAIT_FRAMES: int = 60
TRAINING_WINDOW_MOVE_STEP: int = 4
TRAINING_ROW_MOVE_STEP: int = 80

# ==================== Data Augmentation ====================
SVM_ADD_LEFT_RIGHT_FLIPS: bool = False  # Important for single-hand orientation
SVM_C_PARAMETER: float = 8  # Regularization parameter

# ==================== Video Processing ====================
SCALE_FACTOR: int = 4  # For downsampling frames for faster processing
OUTPUT_VIDEO_FORMAT: str = "mp4"

# ==================== Display Settings ====================
DISPLAY_FPS: bool = True
DISPLAY_CONFIDENCE: bool = True
DISPLAY_HAND_SIZE: bool = True
DISPLAY_HAND_CENTER: bool = True
DISPLAY_WINDOW_NAME: str = "Gesture Recognition System"

# ==================== Keyboard Actions ====================
GESTURE_ACTIONS: Dict[str, tuple] = {
    "Pause": ("press", "space"),
    "Scrolling Up": ("scroll", -7),
    "Scrolling Tabs": ("hotkey", ("ctrl", "pgup")),
    "Change Program": ("hotkey", ("alt", "tab")),
}

# ==================== Logging ====================
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = str(PROJECT_ROOT / "gesture_recognition.log")

# ==================== Processing ====================
FPS_UPDATE_INTERVAL: int = 10  # Update FPS display every N frames
COOLDOWN_MS: int = 500  # Cooldown after performing action (ms)

# ==================== Development ====================
DEBUG_MODE: bool = False
VERBOSE: bool = True

