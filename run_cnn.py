"""
Real-time Gesture Recognition — CNN only
=========================================
Gestures recognised:
  A  →  Pause / Play      (Space bar)
  C  →  Scroll Up
  D  →  Scroll Down
  E  →  Switch Program    (Alt + Tab)

Usage:
  python run_cnn.py                  # default camera
  python run_cnn.py --camera 1       # external webcam
  python run_cnn.py --model path/to/model.h5

Controls (while running):
  Q  — quit
  P  — pause / resume predictions
  R  — re-calibrate background

Requirements:
  pip install opencv-python tensorflow numpy pyautogui
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyautogui
from tensorflow import keras

# ─────────────────────────────────────────────────────────────────────────────
# Settings (override via CLI — no need to edit this file)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import config
    _DEFAULT_MODEL   = config.CNN_MODEL_PATH
    _DEFAULT_CAMERA  = config.CAMERA_ID
    _IMAGE_SIZE      = config.CNN_IMAGE_SIZE        # (W, H)
    _GESTURE_NAMES   = config.CNN_GESTURE_NAMES     # ["A","C","D","E"]
    _GESTURE_ACTIONS = config.CNN_GESTURE_ACTIONS   # {"A": ("press","space"), …}
    _GESTURE_LABELS  = config.CNN_GESTURE_LABELS
    _CONFIDENCE_THR  = config.CNN_CONFIDENCE_THRESHOLD
    _COOLDOWN_MS     = config.COOLDOWN_MS
except ImportError:
    _DEFAULT_MODEL   = "models/gesture_cnn_model.h5"
    _DEFAULT_CAMERA  = 0
    _IMAGE_SIZE      = (300, 300)
    _GESTURE_NAMES   = ["A", "C", "D", "E"]
    _GESTURE_ACTIONS = {
        "A": ("press",  "space"),
        "C": ("scroll",  7),
        "D": ("scroll", -7),
        "E": ("hotkey", "alt", "tab"),
    }
    _GESTURE_LABELS  = {
        "A": "A  →  Pause / Play",
        "C": "C  →  Scroll Up",
        "D": "D  →  Scroll Down",
        "E": "E  →  Switch Program",
    }
    _CONFIDENCE_THR  = 0.90
    _COOLDOWN_MS     = 600

# ── ROI rectangle drawn on the camera frame ──────────────────────────────────
ROI = dict(top=80, bottom=420, left=420, right=760)   # pixels

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
C_GREEN  = (50,  205,  50)
C_RED    = (50,   50, 230)
C_ORANGE = (30,  165, 255)
C_YELLOW = (0,   215, 255)
C_CYAN   = (255, 210,   0)
C_WHITE  = (255, 255, 255)
C_DARK   = (20,   20,  20)

pyautogui.FAILSAFE = False   # prevent corner-crash

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _txt(img, text, pos, scale=0.65, color=C_WHITE, thickness=2):
    """Text with drop-shadow for readability."""
    cv2.putText(img, text, (pos[0]+1, pos[1]+1),
                cv2.FONT_HERSHEY_SIMPLEX, scale, C_DARK, thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color,  thickness,   cv2.LINE_AA)


def _panel(img, x1, y1, x2, y2, alpha=0.6):
    """Semi-transparent dark panel."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), C_DARK, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _confidence_bar(img, x, y, w, h, value, color):
    """Horizontal confidence bar."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
    cv2.rectangle(img, (x, y), (x + int(w * value), y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), C_WHITE, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Action executor
# ─────────────────────────────────────────────────────────────────────────────

class ActionExecutor:
    """Debounced action executor — prevents repeated triggers."""

    def __init__(self, cooldown_ms: int = _COOLDOWN_MS):
        self.cooldown_ms   = cooldown_ms
        self._last: dict   = {}
        self.last_action   = ""
        self.action_time   = 0.0

    def execute(self, gesture: str) -> bool:
        """
        Execute the action for *gesture* if the cooldown has expired.
        Returns True when the action was actually fired.
        """
        now = time.time() * 1000
        if now - self._last.get(gesture, 0) < self.cooldown_ms:
            return False

        action = _GESTURE_ACTIONS.get(gesture)
        if action is None:
            return False

        kind, *params = action
        try:
            if kind == "press":
                pyautogui.press(params[0])
            elif kind == "scroll":
                pyautogui.scroll(params[0])
            elif kind == "hotkey":
                pyautogui.hotkey(*params)
        except Exception as exc:
            log.error(f"Action failed for {gesture}: {exc}")
            return False

        self._last[gesture] = now
        self.last_action    = _GESTURE_LABELS.get(gesture, gesture)
        self.action_time    = time.time()
        log.info(f"Action fired ▶ {self.last_action}")
        return True


# ─────────────────────────────────────────────────────────────────────────────
# CNN predictor
# ─────────────────────────────────────────────────────────────────────────────

class CNNPredictor:
    """Wraps a Keras model and provides a single predict() call."""

    def __init__(self, model_path: str):
        if not Path(model_path).exists():
            log.error(f"Model not found: {model_path}")
            log.error("Train your model first:  python train_cnn.py")
            sys.exit(1)

        log.info(f"Loading model from {model_path} …")
        self.model = keras.models.load_model(model_path)
        log.info("Model loaded ✓")
        self.model.summary(print_fn=lambda x: log.debug(x))

    def predict(self, roi_gray: np.ndarray):
        """
        Parameters
        ----------
        roi_gray : grayscale image crop (any size — will be resized internally)

        Returns
        -------
        gesture : str   — e.g. "A"
        confidence : float   — 0.0 … 1.0
        all_probs : np.ndarray  — probability per class
        """
        img = cv2.resize(roi_gray, _IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        img = img.astype("float32") / 255.0                  # normalise 0-1
        img = img.reshape(1, _IMAGE_SIZE[1], _IMAGE_SIZE[0], 1)  # (1, H, W, 1)

        probs     = self.model.predict(img, verbose=0)[0]
        idx       = int(np.argmax(probs))
        gesture   = _GESTURE_NAMES[idx]
        confidence = float(probs[idx])

        return gesture, confidence, probs


# ─────────────────────────────────────────────────────────────────────────────
# Main application
# ─────────────────────────────────────────────────────────────────────────────

class GestureApp:
    def __init__(self, model_path: str, camera_id: int, confidence_thr: float):
        self.predictor   = CNNPredictor(model_path)
        self.executor    = ActionExecutor()
        self.camera_id   = camera_id
        self.conf_thr    = confidence_thr

        self.paused      = False
        self.running     = True
        self.fps         = 0.0
        self._fc         = 0
        self._t0         = time.time()

        # prediction smoothing (majority vote over last N frames)
        self._smooth_n   = 5
        self._history: list = []

    # ── FPS ───────────────────────────────────────────────────────────────────
    def _update_fps(self):
        self._fc += 1
        if self._fc % 20 == 0:
            self.fps = self._fc / (time.time() - self._t0)

    # ── Prediction smoothing (simple majority vote) ───────────────────────────
    def _smooth(self, gesture: str, confidence: float):
        """Returns the smoothed gesture if confidence is stable."""
        self._history.append((gesture, confidence))
        if len(self._history) > self._smooth_n:
            self._history.pop(0)

        # Majority vote among recent frames
        from collections import Counter
        votes    = Counter(g for g, _ in self._history)
        top, cnt = votes.most_common(1)[0]
        avg_conf = np.mean([c for g, c in self._history if g == top])

        if cnt >= self._smooth_n // 2 + 1:
            return top, float(avg_conf)
        return None, 0.0

    # ── UI drawing ────────────────────────────────────────────────────────────
    def _draw(self, frame: np.ndarray, gesture: str, confidence: float,
              all_probs: np.ndarray, stable: str, stable_conf: float) -> np.ndarray:
        h, w = frame.shape[:2]

        # ── ROI rectangle ──────────────────────────────────────────────────
        roi_color = C_ORANGE if self.paused else (C_GREEN if stable else C_RED)
        cv2.rectangle(frame,
                      (ROI["left"],  ROI["top"]),
                      (ROI["right"], ROI["bottom"]),
                      roi_color, 3)
        _txt(frame, "HAND ROI",
             (ROI["left"] + 8, ROI["top"] - 10),
             scale=0.5, color=roi_color)

        # ── Top info panel ─────────────────────────────────────────────────
        _panel(frame, 0, 0, 400, 155)

        _txt(frame, "CNN Gesture Recognition",
             (10, 28), scale=0.7, color=C_CYAN, thickness=2)

        status = "PAUSED (P to resume)" if self.paused else "RUNNING"
        s_col  = C_YELLOW if self.paused else C_GREEN
        _txt(frame, status, (10, 56), scale=0.55, color=s_col)

        _txt(frame, f"FPS: {self.fps:.1f}", (10, 80), scale=0.55, color=C_WHITE)

        # Live (unsmoothed) prediction
        raw_label = _GESTURE_LABELS.get(gesture, gesture)
        _txt(frame, f"Detected: {raw_label}", (10, 104),
             scale=0.55, color=C_WHITE)
        _confidence_bar(frame, 10, 112, 230, 12, confidence,
                        C_GREEN if confidence >= self.conf_thr else C_RED)
        _txt(frame, f"{confidence*100:.1f}%", (248, 122),
             scale=0.45, color=C_WHITE, thickness=1)

        # ── Stable gesture (big) ───────────────────────────────────────────
        if stable:
            _panel(frame, 0, h - 100, w, h)
            label = _GESTURE_LABELS.get(stable, stable)
            _txt(frame, label, (20, h - 62),
                 scale=1.1, color=C_CYAN, thickness=2)
            _txt(frame, f"confidence {stable_conf*100:.0f}%",
                 (20, h - 30), scale=0.6, color=C_WHITE)

        # ── Last action banner ─────────────────────────────────────────────
        if self.executor.last_action and (time.time() - self.executor.action_time) < 1.5:
            (tw, _), _ = cv2.getTextSize(
                "✓ " + self.executor.last_action,
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            bx = (w - tw) // 2
            _panel(frame, bx - 12, h - 150, bx + tw + 12, h - 112)
            _txt(frame, "✓ " + self.executor.last_action,
                 (bx, h - 122), scale=0.75, color=C_GREEN, thickness=2)

        # ── Per-class probability bars (right side) ────────────────────────
        px = w - 220
        _panel(frame, px - 8, 0, w, len(_GESTURE_NAMES) * 50 + 20)
        _txt(frame, "Class probabilities", (px, 20),
             scale=0.45, color=C_WHITE, thickness=1)
        for i, (gname, prob) in enumerate(zip(_GESTURE_NAMES, all_probs)):
            y0 = 30 + i * 48
            _txt(frame, _GESTURE_LABELS.get(gname, gname),
                 (px, y0 + 14), scale=0.42, color=C_CYAN, thickness=1)
            bar_col = C_GREEN if gname == stable else C_WHITE
            _confidence_bar(frame, px, y0 + 18, 200, 16, float(prob), bar_col)
            _txt(frame, f"{prob*100:.0f}%",
                 (px + 205, y0 + 30), scale=0.42, color=C_WHITE, thickness=1)

        # ── Controls ──────────────────────────────────────────────────────
        _panel(frame, 0, h - (50 if stable else 50) - 60,
               230, h - (50 if stable else 50))
        for i, (k, d) in enumerate([("[P]", "Pause/Resume"),
                                     ("[R]", "Re-calibrate BG"),
                                     ("[Q]", "Quit")]):
            _txt(frame, f"{k} {d}",
                 (8, h - (50 if stable else 50) - 50 + i * 18),
                 scale=0.42, color=C_YELLOW, thickness=1)

        return frame

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            # try fallback cameras
            for cid in [0, 1, 2]:
                if cid == self.camera_id:
                    continue
                cap = cv2.VideoCapture(cid)
                if cap.isOpened():
                    log.warning(f"Camera {self.camera_id} failed — using {cid}")
                    break
            else:
                log.error("No camera found!")
                return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        cv2.namedWindow("Gesture Recognition – CNN", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Gesture Recognition – CNN", 1100, 680)

        log.info("Running — press Q to quit, P to pause, R to re-calibrate")

        # Placeholders so we always have something to draw
        gesture, confidence, all_probs = _GESTURE_NAMES[0], 0.0, np.zeros(len(_GESTURE_NAMES))
        stable, stable_conf = None, 0.0

        while self.running:
            ret, frame = cap.read()
            if not ret:
                log.warning("Frame read error — retrying…")
                time.sleep(0.03)
                continue

            frame = cv2.flip(frame, 1)         # mirror
            self._update_fps()

            # ── Extract ROI and convert to grayscale ──────────────────────
            roi_bgr  = frame[ROI["top"]:ROI["bottom"],
                             ROI["left"]:ROI["right"]]
            roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

            # ── Predict (every frame, unless paused) ──────────────────────
            if not self.paused and roi_gray.size > 0:
                gesture, confidence, all_probs = self.predictor.predict(roi_gray)
                stable, stable_conf = self._smooth(gesture, confidence)

                # ── Trigger action if confidence is high enough ────────────
                if stable and stable_conf >= self.conf_thr:
                    self.executor.execute(stable)

            # ── Draw UI ───────────────────────────────────────────────────
            display = self._draw(frame.copy(), gesture, confidence,
                                 all_probs, stable, stable_conf)

            # ── ROI preview thumbnail (top-right corner inset) ────────────
            thumb = cv2.resize(roi_bgr, (130, 130))
            fh, fw = display.shape[:2]
            display[12:142, fw - 142:fw - 12] = thumb
            cv2.rectangle(display, (fw - 142, 12), (fw - 12, 142), C_WHITE, 2)
            _txt(display, "ROI preview", (fw - 142, 10),
                 scale=0.38, color=C_WHITE, thickness=1)

            cv2.imshow("Gesture Recognition – CNN", display)

            # ── Key handling ──────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                log.info("Quit requested.")
                break
            elif key == ord('p'):
                self.paused = not self.paused
                log.info("Paused" if self.paused else "Resumed")
            elif key == ord('r'):
                self._history.clear()
                log.info("History cleared — predictions reset")

        cap.release()
        cv2.destroyAllWindows()
        log.info("Session ended.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Real-time CNN gesture recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Gesture → Action mapping:
  A  →  Space bar  (Pause / Play)
  C  →  Scroll Up
  D  →  Scroll Down
  E  →  Alt + Tab  (Switch Program)

Examples:
  python run_cnn.py
  python run_cnn.py --camera 1
  python run_cnn.py --model models/gesture_cnn_model.h5 --confidence 0.85
        """
    )
    p.add_argument("--model",      default=_DEFAULT_MODEL,
                   help=f"Path to .h5 model (default: {_DEFAULT_MODEL})")
    p.add_argument("--camera",     type=int, default=_DEFAULT_CAMERA,
                   help=f"Camera device ID (default: {_DEFAULT_CAMERA})")
    p.add_argument("--confidence", type=float, default=_CONFIDENCE_THR,
                   help=f"Confidence threshold 0-1 (default: {_CONFIDENCE_THR})")
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "═" * 60)
    print("  🖐  GESTURE RECOGNITION  —  CNN Mode")
    print("═" * 60)
    print(f"  Model      : {args.model}")
    print(f"  Camera ID  : {args.camera}")
    print(f"  Confidence : {args.confidence * 100:.0f} %  minimum")
    print("─" * 60)
    print("  Gesture  │  Action")
    print("  ─────────┼──────────────────────────")
    for g, lbl in _GESTURE_LABELS.items():
        print(f"  {lbl}")
    print("─" * 60)
    print("  [P] pause   [R] reset   [Q] quit")
    print("═" * 60 + "\n")

    GestureApp(
        model_path     = args.model,
        camera_id      = args.camera,
        confidence_thr = args.confidence,
    ).run()


if __name__ == "__main__":
    main()

