#!/usr/bin/env python3
"""
Quick start script for Gesture Recognition Application.
This script helps users get started quickly with the application.
"""

import os
import sys
from pathlib import Path


def print_header():
    """Print welcome header."""
    print("\n" + "=" * 80)
    print("GESTURE RECOGNITION APPLICATION - QUICK START")
    print("=" * 80 + "\n")


def print_menu():
    """Print main menu."""
    print("What would you like to do?\n")
    print("  1. Run Real-time Gesture Recognition (SVM)")
    print("  2. Run Hand Segmentation Mode (Finger Counting)")
    print("  3. View Configuration")
    print("  4. Install Dependencies")
    print("  5. View Documentation")
    print("  6. Exit\n")


def check_dependencies():
    """Check if all dependencies are installed."""
    required_packages = [
        'cv2',
        'tensorflow',
        'sklearn',
        'dlib',
        'pyautogui',
        'numpy',
    ]

    print("\nChecking dependencies...")
    missing = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package}")
            missing.append(package)

    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("\nRun option 4 to install missing dependencies.\n")
        return False

    print("\n✓ All dependencies installed!\n")
    return True


def install_dependencies():
    """Install dependencies from requirements.txt."""
    print("\nInstalling dependencies...")

    requirements_file = Path(__file__).parent / "requirements.txt"

    if not requirements_file.exists():
        print(f"ERROR: requirements.txt not found at {requirements_file}")
        return False

    import subprocess

    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("\n✓ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error installing dependencies: {str(e)}")
        return False


def run_gesture_recognition():
    """Run gesture recognition application."""
    print("\nStarting Gesture Recognition (SVM Mode)...")
    print("Press 'q' to quit\n")

    try:
        os.system("python app.py --mode svm --camera 0")
    except Exception as e:
        print(f"Error: {str(e)}")


def run_hand_segmentation():
    """Run hand segmentation application."""
    print("\nStarting Hand Segmentation Mode...")
    print("Press 'q' to quit")
    print("Please position your hand in the ROI (region of interest)\n")

    try:
        os.system("python app.py --mode segment --camera 0")
    except Exception as e:
        print(f"Error: {str(e)}")


def view_configuration():
    """Display configuration options."""
    print("\n" + "=" * 80)
    print("CONFIGURATION OVERVIEW")
    print("=" * 80 + "\n")

    print("Edit 'config.py' to customize:\n")

    print("Camera Settings:")
    print("  - CAMERA_ID: which camera to use (0=default, 1=external)")
    print("  - CAMERA_WIDTH, CAMERA_HEIGHT: frame resolution")
    print("  - SCALE_FACTOR: downsampling (higher = faster)")

    print("\nGesture Actions:")
    print("  - GESTURE_ACTIONS: map gestures to keyboard/mouse actions")
    print("  - SVM_CONFIDENCE_THRESHOLD: detection sensitivity")
    print("  - COOLDOWN_MS: time between actions (prevents spam)")

    print("\nDisplay Options:")
    print("  - DISPLAY_FPS: show FPS counter")
    print("  - DISPLAY_CONFIDENCE: show detection confidence")
    print("  - DISPLAY_HAND_SIZE: show hand size")

    print("\nModel Settings:")
    print("  - CNN_EPOCHS: training epochs")
    print("  - CNN_BATCH_SIZE: batch size for training")
    print("  - CNN_IMAGE_SIZE: input image resolution")

    print("\n" + "=" * 80 + "\n")


def view_documentation():
    """View documentation files."""
    print("\nDocumentation Files:\n")

    docs = {
        "README_MODERNIZED.md": "Complete user guide and API reference",
        "MODERNIZATION_REPORT.md": "Detailed analysis of improvements",
        "config.py": "Configuration file with all customizable parameters",
    }

    for filename, description in docs.items():
        print(f"  • {filename}")
        print(f"    {description}\n")

    print("View files in your editor or with:")
    print("  - cat README_MODERNIZED.md")
    print("  - less MODERNIZATION_REPORT.md\n")


def main():
    """Main entry point."""
    print_header()

    # Check if we're in the right directory
    if not Path("app.py").exists():
        print("ERROR: This script must be run from the gesture-recognition-actions directory")
        print("Please navigate to the project directory and try again.")
        sys.exit(1)

    while True:
        print_menu()

        choice = input("Enter your choice (1-6): ").strip()

        if choice == "1":
            if check_dependencies():
                run_gesture_recognition()

        elif choice == "2":
            if check_dependencies():
                run_hand_segmentation()

        elif choice == "3":
            view_configuration()

        elif choice == "4":
            if install_dependencies():
                pass

        elif choice == "5":
            view_documentation()

        elif choice == "6":
            print("\nGoodbye! 👋\n")
            sys.exit(0)

        else:
            print("\nInvalid choice. Please try again.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye! 👋\n")
        sys.exit(0)

