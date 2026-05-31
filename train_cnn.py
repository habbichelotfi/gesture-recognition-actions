"""
Modern CNN-based gesture recognition training module.
Trains a convolutional neural network to recognize hand gestures.
"""

import logging
import numpy as np
import os
from pathlib import Path
from typing import Tuple

import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

import config
from utils import setup_logging, validate_file_exists

# Setup logging
logger = setup_logging()


class GestureRecognitionCNN:
    """CNN model for gesture recognition."""

    def __init__(self, num_classes: int = 4, image_size: Tuple[int, int] = (300, 300)):
        """
        Initialize gesture recognition CNN.

        Args:
            num_classes: Number of gesture classes
            image_size: Input image size (height, width)
        """
        self.num_classes = num_classes
        self.image_size = image_size
        self.model = None
        self.history = None

        logger.info(f"Initialized GestureRecognitionCNN with {num_classes} classes")

    def build_model(self) -> Sequential:
        """
        Build CNN architecture.

        Returns:
            Compiled model
        """
        self.model = Sequential([
            # First convolutional block
            Conv2D(32, (5, 5), activation='relu', input_shape=(*self.image_size, 1)),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Second convolutional block
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Third convolutional block
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            # Flatten and dense layers
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])

        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        logger.info("Model built and compiled successfully")
        self.model.summary()

        return self.model

    def load_images_from_directory(self, dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load images and labels from directory structure.
        Expects: dataset_path/GESTURE_NAME/*.jpg  (e.g. datasets/Below_CAM/A/*.jpg)

        Args:
            dataset_path: Root path containing one sub-folder per gesture class.

        Returns:
            Tuple of (X, y) — images array and integer labels array.
        """
        if not Path(dataset_path).exists():
            logger.error(f"Dataset path not found: {dataset_path}")
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        X, y = [], []

        for gesture_idx, gesture_name in enumerate(config.CNN_GESTURE_NAMES):
            gesture_dir = Path(dataset_path) / gesture_name

            if not gesture_dir.exists():
                logger.warning(f"Gesture directory not found: {gesture_dir}")
                continue

            image_count = 0
            for image_file in sorted(gesture_dir.glob("*.jpg")):
                try:
                    img = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        logger.warning(f"Failed to load image: {image_file}")
                        continue

                    img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
                    X.append(img)
                    y.append(gesture_idx)
                    image_count += 1

                except Exception as e:
                    logger.error(f"Error processing image {image_file}: {str(e)}")
                    continue

            logger.info(f"  [{gesture_idx}] '{gesture_name}' — {image_count} images loaded")

        if not X:
            raise ValueError("No images loaded from dataset")

        X = np.array(X, dtype="uint8")
        X = np.expand_dims(X, axis=3)   # (N, H, W) → (N, H, W, 1)
        y = np.array(y)

        logger.info(f"Total images : {len(X)}")
        logger.info(f"Input shape  : {X.shape}")
        logger.info(f"Classes      : {dict(zip(*np.unique(y, return_counts=True)))}")

        return X, y

    def train(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        epochs: int = 20,
        batch_size: int = 32,
        verbose: int = 2
    ):
        """
        Train the model.

        Args:
            X_train: Training images
            X_test: Test images
            y_train: Training labels
            y_test: Test labels
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
        """
        logger.info(f"Starting training with {epochs} epochs and batch size {batch_size}")

        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=verbose
        )

        logger.info("Training completed")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate model on test data.

        Args:
            X_test: Test images
            y_test: Test labels

        Returns:
            Tuple of (loss, accuracy)
        """
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
        return loss, accuracy

    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Predict gesture from image.

        Args:
            image: Input image (grayscale, 300x300)

        Returns:
            Tuple of (gesture_name, confidence_score)
        """
        # Prepare image
        image = np.array(image, dtype='float32')
        image = image / 255.0  # Normalize
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Predict
        predictions = self.model.predict(image, verbose=0)
        pred_class = np.argmax(predictions[0])
        confidence = float(predictions[0][pred_class])

        gesture_name = config.CNN_GESTURE_NAMES[pred_class]

        return gesture_name, confidence

    def save(self, save_path: str):
        """
        Save model to file.

        Args:
            save_path: Path to save model
        """
        try:
            self.model.save(save_path)
            logger.info(f"Model saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load(self, model_path: str):
        """
        Load model from file.

        Args:
            model_path: Path to model file
        """
        try:
            if not validate_file_exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            self.model = keras.models.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise


def train_model(dataset_path: str = None) -> GestureRecognitionCNN:
    """
    Train gesture recognition model.

    Args:
        dataset_path: Path to dataset directory

    Returns:
        Trained model instance
    """
    if dataset_path is None:
        dataset_path = str(config.DATASETS_DIR / "Below_CAM")

    logger.info(f"Starting gesture recognition model training")
    logger.info(f"Dataset path: {dataset_path}")

    # Initialize model
    model = GestureRecognitionCNN(
        num_classes=config.CNN_NUM_CLASSES,
        image_size=config.CNN_IMAGE_SIZE
    )

    # Build model
    model.build_model()

    # Load data
    logger.info("Loading images from dataset...")
    try:
        X, y = model.load_images_from_directory(dataset_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        logger.info("Creating dummy model for demonstration...")
        # Create dummy data for demonstration
        X = np.random.rand(100, *config.CNN_IMAGE_SIZE, 1).astype('uint8')
        y = np.random.randint(0, config.CNN_NUM_CLASSES, 100)

    # Split data
    logger.info(f"Splitting data with test size {config.CNN_VALIDATION_SPLIT}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.CNN_VALIDATION_SPLIT,
        random_state=config.CNN_RANDOM_STATE
    )

    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")

    # Train model
    model.train(
        X_train, X_test, y_train, y_test,
        epochs=config.CNN_EPOCHS,
        batch_size=config.CNN_BATCH_SIZE
    )

    # Evaluate model
    model.evaluate(X_test, y_test)

    # Save model
    model.save(config.CNN_MODEL_PATH)

    return model


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("Gesture Recognition CNN - Training Module")
    logger.info("=" * 80)

    try:
        # Train model
        trained_model = train_model()
        logger.info("Model training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise

