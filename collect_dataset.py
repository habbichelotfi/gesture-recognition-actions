"""
╔══════════════════════════════════════════════════════════════╗
║         GESTURE DATASET COLLECTOR - Interactive Tool         ║
║                                                              ║
║  Easily capture hand gesture images to build your dataset.   ║
║  Images are auto-organized into the correct folder structure  ║
║  for training with train_cnn.py                              ║
╚══════════════════════════════════════════════════════════════╝

Controls:
  [SPACE] ──── Capture single image
  [HOLD H] ─── Auto-capture (burst mode)
  [N] ──────── Next gesture class
  [P] ──────── Previous gesture class
  [A] ──────── Add a new custom gesture class
  [D] ──────── Delete last captured image
  [C] ──────── Clear all images for current class
  [S] ──────── Show dataset stats
  [Q] ──────── Quit and save session report
"""

import cv2
import os
import time
import json
import shutil
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
try:
    import config
    DATASETS_DIR    = config.DATASETS_DIR / "Below_CAM"
    CAMERA_ID       = config.CAMERA_ID
    CAMERA_WIDTH    = config.CAMERA_WIDTH
    CAMERA_HEIGHT   = config.CAMERA_HEIGHT
    GESTURE_NAMES   = list(config.CNN_GESTURE_NAMES)
except ImportError:
    DATASETS_DIR    = Path("datasets/Below_CAM")
    CAMERA_ID       = 0
    CAMERA_WIDTH    = 1280
    CAMERA_HEIGHT   = 720
    GESTURE_NAMES   = ["A", "C", "D", "E", "F", "G", "H", "K",
                       "L", "N", "P", "Q", "S", "U", "V", "W", "Z"]

# ── Capture settings ──────────────────────────────────────────────────────────
IMAGE_SIZE           = (300, 300)      # Size images are saved at (W, H)
AUTO_CAPTURE_DELAY   = 0.15            # Seconds between auto-captures (hold H)
CAPTURE_COUNTDOWN    = 3              # Seconds countdown before burst starts
MIN_IMAGES_PER_CLASS = 100            # Warning threshold
TARGET_IMAGES        = 200            # Target goal shown in UI

# ── ROI (Region Of Interest) ─────────────────────────────────────────────────
ROI_TOP    = 80
ROI_LEFT   = 420
ROI_BOTTOM = 420
ROI_RIGHT  = 760

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
COL_GREEN    = (50,  205,  50)
COL_RED      = (60,   60, 255)
COL_ORANGE   = (30,  165, 255)
COL_YELLOW   = (0,   215, 255)
COL_WHITE    = (255, 255, 255)
COL_DARK     = (20,   20,  20)
COL_CYAN     = (255, 220,   0)
COL_PURPLE   = (200,  60, 200)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def count_images(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for f in directory.iterdir()
               if f.suffix.lower() in (".jpg", ".jpeg", ".png"))


def draw_rounded_rect(img, pt1, pt2, color, thickness=2, radius=12, filled=False):
    """Draw a rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    if filled:
        overlay = img.copy()
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                       (x1+radius, y2-radius), (x2-radius, y2-radius)]:
            cv2.circle(overlay, (cx, cy), radius, color, -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    else:
        cv2.line(img, (x1+radius, y1), (x2-radius, y1), color, thickness)
        cv2.line(img, (x1+radius, y2), (x2-radius, y2), color, thickness)
        cv2.line(img, (x1, y1+radius), (x1, y2-radius), color, thickness)
        cv2.line(img, (x2, y1+radius), (x2, y2-radius), color, thickness)
        cv2.ellipse(img, (x1+radius, y1+radius), (radius,radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-radius, y1+radius), (radius,radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1+radius, y2-radius), (radius,radius),  90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-radius, y2-radius), (radius,radius),   0, 0, 90, color, thickness)


def put_text_shadow(img, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX,
                    scale=0.7, color=COL_WHITE, thickness=2):
    """Draw text with a dark drop-shadow for readability."""
    x, y = pos
    cv2.putText(img, text, (x+1, y+1), font, scale, COL_DARK, thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y),     font, scale, color,    thickness,   cv2.LINE_AA)


def progress_bar(img, x, y, w, h, progress, color=COL_GREEN, bg=(60, 60, 60)):
    """Draw a progress bar."""
    cv2.rectangle(img, (x, y), (x+w, y+h), bg, -1)
    filled = int(w * min(progress, 1.0))
    if filled > 0:
        cv2.rectangle(img, (x, y), (x+filled, y+h), color, -1)
    cv2.rectangle(img, (x, y), (x+w, y+h), COL_WHITE, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Session report
# ─────────────────────────────────────────────────────────────────────────────

class SessionReport:
    def __init__(self):
        self.start_time  = datetime.now()
        self.captures: Dict[str, int] = {}
        self.deleted:  Dict[str, int] = {}

    def add_capture(self, gesture: str):
        self.captures[gesture] = self.captures.get(gesture, 0) + 1

    def add_delete(self, gesture: str):
        self.deleted[gesture] = self.deleted.get(gesture, 0) + 1

    def save(self, path: Path):
        report = {
            "session_start": self.start_time.isoformat(),
            "session_end":   datetime.now().isoformat(),
            "total_captured": sum(self.captures.values()),
            "total_deleted":  sum(self.deleted.values()),
            "per_gesture":    {g: {"captured": self.captures.get(g, 0),
                                   "deleted":  self.deleted.get(g, 0)}
                               for g in set(list(self.captures) + list(self.deleted))}
        }
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Session report saved → {path}")
        return report


# ─────────────────────────────────────────────────────────────────────────────
# Main Collector
# ─────────────────────────────────────────────────────────────────────────────

class DatasetCollector:
    def __init__(self, gestures: List[str], dataset_dir: Path,
                 camera_id: int = 0):
        self.gestures     = gestures
        self.dataset_dir  = dataset_dir
        self.camera_id    = camera_id
        self.current_idx  = 0

        self.cap                 = None
        self.auto_capturing      = False
        self.last_capture_time   = 0.0
        self.countdown_start     = None
        self.flash_alpha         = 0.0        # capture flash effect
        self.session             = SessionReport()
        self.status_msg          = ""
        self.status_color        = COL_GREEN
        self.status_expire       = 0.0

        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Dataset directory: {self.dataset_dir}")

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def current_gesture(self) -> str:
        return self.gestures[self.current_idx]

    @property
    def current_gesture_dir(self) -> Path:
        d = self.dataset_dir / self.current_gesture
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def current_count(self) -> int:
        return count_images(self.current_gesture_dir)

    # ── Camera ────────────────────────────────────────────────────────────────

    def open_camera(self) -> bool:
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            # Try fallback camera IDs
            for cid in [0, 1, 2]:
                if cid == self.camera_id:
                    continue
                self.cap = cv2.VideoCapture(cid)
                if self.cap.isOpened():
                    logger.warning(f"Camera {self.camera_id} failed — using camera {cid}")
                    break
            else:
                logger.error("No camera found!")
                return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        logger.info("Camera opened successfully")
        return True

    def release(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    # ── Capture ───────────────────────────────────────────────────────────────

    def _next_filename(self) -> Path:
        """Return a unique filename using timestamp."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return self.current_gesture_dir / f"{self.current_gesture}_{ts}.jpg"

    def capture_image(self, roi: np.ndarray) -> bool:
        """Save the ROI image to disk."""
        try:
            img_save = cv2.resize(roi, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            gray     = cv2.cvtColor(img_save, cv2.COLOR_BGR2GRAY)
            filepath = self._next_filename()
            cv2.imwrite(str(filepath), gray)
            self.session.add_capture(self.current_gesture)
            self.flash_alpha = 1.0
            self._set_status(f"✓ Saved [{self.current_count}]", COL_GREEN, 1.2)
            return True
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            self._set_status(f"✗ Save error: {e}", COL_RED, 2.0)
            return False

    def delete_last(self):
        """Delete the most recently captured image."""
        files = sorted(self.current_gesture_dir.glob("*.jpg"),
                       key=os.path.getmtime)
        if files:
            files[-1].unlink()
            self.session.add_delete(self.current_gesture)
            self._set_status(f"🗑  Deleted last image [{self.current_count} remain]",
                             COL_ORANGE, 1.5)
        else:
            self._set_status("No images to delete", COL_YELLOW, 1.5)

    def clear_class(self):
        """Delete ALL images for the current gesture."""
        count = self.current_count
        if count == 0:
            self._set_status("No images to clear", COL_YELLOW, 1.5)
            return
        shutil.rmtree(str(self.current_gesture_dir))
        self.current_gesture_dir.mkdir(parents=True, exist_ok=True)
        self._set_status(f"🗑  Cleared {count} images for '{self.current_gesture}'",
                         COL_RED, 2.0)

    # ── Status ────────────────────────────────────────────────────────────────

    def _set_status(self, msg: str, color=COL_GREEN, duration: float = 1.5):
        self.status_msg    = msg
        self.status_color  = color
        self.status_expire = time.time() + duration

    def _get_status(self):
        if time.time() < self.status_expire:
            return self.status_msg, self.status_color
        return "", COL_WHITE

    # ── Dataset stats ─────────────────────────────────────────────────────────

    def print_stats(self):
        print("\n" + "═" * 55)
        print(f"  📊  DATASET STATISTICS  — {self.dataset_dir}")
        print("═" * 55)
        total = 0
        for g in self.gestures:
            n    = count_images(self.dataset_dir / g)
            bar  = "█" * (n // 5) + "░" * max(0, TARGET_IMAGES // 5 - n // 5)
            flag = "✅" if n >= MIN_IMAGES_PER_CLASS else "⚠️ "
            print(f"  {flag} {g:>4}  {bar[:40]}  {n:>4} imgs")
            total += n
        print("═" * 55)
        print(f"  Total images: {total}")
        print("═" * 55 + "\n")

    # ── UI Drawing ────────────────────────────────────────────────────────────

    def _draw_ui(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        now   = time.time()

        # ── ROI rectangle ──────────────────────────────────────────────────
        roi_color = COL_GREEN if not self.auto_capturing else COL_ORANGE
        cv2.rectangle(frame,
                      (ROI_LEFT,  ROI_TOP),
                      (ROI_RIGHT, ROI_BOTTOM),
                      roi_color, 3)
        put_text_shadow(frame, "PLACE HAND HERE",
                        (ROI_LEFT + 10, ROI_TOP - 10),
                        scale=0.55, color=roi_color)

        # ── Top header panel ───────────────────────────────────────────────
        draw_rounded_rect(frame, (10, 8), (w - 10, 72),
                          COL_DARK, filled=True, radius=10)

        # Gesture name (big)
        put_text_shadow(frame, f"Gesture: {self.current_gesture}",
                        (20, 50), scale=1.2, color=COL_CYAN,
                        thickness=2)

        # Class index
        idx_txt = f"{self.current_idx + 1}/{len(self.gestures)}"
        put_text_shadow(frame, idx_txt, (w - 130, 50), scale=0.9,
                        color=COL_WHITE)

        # ── Progress panel ─────────────────────────────────────────────────
        panel_y = 82
        draw_rounded_rect(frame, (10, panel_y), (w - 10, panel_y + 52),
                          (30, 30, 30), filled=True, radius=8)

        count  = self.current_count
        pct    = count / TARGET_IMAGES
        bar_w  = w - 120
        bar_col = (COL_GREEN  if pct >= 1.0  else
                   COL_ORANGE if pct >= 0.5  else COL_RED)
        progress_bar(frame, 20, panel_y + 8, bar_w, 20, pct, color=bar_col)
        put_text_shadow(frame, f"{count} / {TARGET_IMAGES} images",
                        (20, panel_y + 48), scale=0.6, color=COL_WHITE)

        if count >= MIN_IMAGES_PER_CLASS:
            put_text_shadow(frame, "✓ READY TO TRAIN",
                            (bar_w - 60, panel_y + 48), scale=0.55,
                            color=COL_GREEN)

        # ── All-class mini-bars on the right ───────────────────────────────
        sidebar_x = w - 175
        sidebar_y = 145
        draw_rounded_rect(frame,
                          (sidebar_x - 6, sidebar_y - 20),
                          (w - 6, sidebar_y + len(self.gestures) * 18 + 4),
                          (25, 25, 25), filled=True, radius=6)
        put_text_shadow(frame, "All classes", (sidebar_x, sidebar_y - 4),
                        scale=0.45, color=COL_WHITE)

        for i, g in enumerate(self.gestures):
            n      = count_images(self.dataset_dir / g)
            ratio  = min(n / TARGET_IMAGES, 1.0)
            bar_len = int(80 * ratio)
            y      = sidebar_y + i * 18 + 14
            col    = (COL_GREEN if n >= MIN_IMAGES_PER_CLASS else
                      COL_ORANGE if n > 0 else (60, 60, 60))
            active = (i == self.current_idx)
            txt_col = COL_CYAN if active else COL_WHITE
            cv2.rectangle(frame, (sidebar_x + 22, y - 10),
                          (sidebar_x + 22 + bar_len, y - 2), col, -1)
            put_text_shadow(frame, f"{g:>3}", (sidebar_x, y - 2),
                            scale=0.4, color=txt_col)
            put_text_shadow(frame, str(n),
                            (sidebar_x + 108, y - 2), scale=0.38,
                            color=txt_col)

        # ── Controls legend ────────────────────────────────────────────────
        legend_y = h - 130
        draw_rounded_rect(frame, (10, legend_y), (380, h - 10),
                          COL_DARK, filled=True, radius=8)
        controls = [
            ("[SPACE]",  "Capture image"),
            ("[HOLD H]", "Auto-capture burst"),
            ("[N] / [P]","Next / Previous class"),
            ("[A]",      "Add custom class"),
            ("[D]",      "Delete last image"),
            ("[C]",      "Clear this class"),
            ("[S]",      "Print stats"),
            ("[Q]",      "Quit & save report"),
        ]
        for i, (key, desc) in enumerate(controls):
            y_pos = legend_y + 14 + i * 15
            put_text_shadow(frame, key,  (18,  y_pos), scale=0.42,
                            color=COL_YELLOW, thickness=1)
            put_text_shadow(frame, desc, (115, y_pos), scale=0.42,
                            color=COL_WHITE,  thickness=1)

        # ── Status message ─────────────────────────────────────────────────
        msg, col = self._get_status()
        if msg:
            (tw, th), _ = cv2.getTextSize(
                msg, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            bx = (w - tw) // 2
            draw_rounded_rect(frame,
                              (bx - 12, h - 160),
                              (bx + tw + 12, h - 132),
                              COL_DARK, filled=True, radius=8)
            put_text_shadow(frame, msg, (bx, h - 140),
                            scale=0.65, color=col, thickness=2)

        # ── Auto-capture countdown ─────────────────────────────────────────
        if self.countdown_start is not None:
            elapsed   = now - self.countdown_start
            remaining = CAPTURE_COUNTDOWN - elapsed
            if remaining > 0:
                txt = f"Starting in {remaining:.1f}s — HOLD STEADY"
                (tw, _), _ = cv2.getTextSize(
                    txt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                put_text_shadow(frame, txt,
                                ((w - tw) // 2, h // 2),
                                scale=0.9, color=COL_ORANGE, thickness=2)
            else:
                self.countdown_start = None
                self.auto_capturing  = True

        # ── Auto-capture indicator ─────────────────────────────────────────
        if self.auto_capturing:
            cv2.circle(frame, (w - 25, 25), 10, COL_RED, -1)
            put_text_shadow(frame, "REC",
                            (w - 65, 30), scale=0.55, color=COL_RED)

        # ── Flash effect ───────────────────────────────────────────────────
        if self.flash_alpha > 0:
            overlay = np.ones_like(frame, dtype=np.uint8) * 255
            cv2.addWeighted(overlay, self.flash_alpha * 0.35,
                            frame,   1 - self.flash_alpha * 0.35,
                            0, frame)
            self.flash_alpha = max(0.0, self.flash_alpha - 0.15)

        return frame

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        if not self.open_camera():
            return

        cv2.namedWindow("Gesture Dataset Collector", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Gesture Dataset Collector", 1100, 720)
        logger.info("Dataset Collector started. Press 'H' to begin auto-capture.")
        self.print_stats()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Frame read failed — retrying…")
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)          # mirror view
            roi   = frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT].copy()

            # ── Auto-capture ────────────────────────────────────────────────
            now = time.time()
            if self.auto_capturing:
                if now - self.last_capture_time >= AUTO_CAPTURE_DELAY:
                    self.capture_image(roi)
                    self.last_capture_time = now

            # ── Draw UI ─────────────────────────────────────────────────────
            display = self._draw_ui(frame.copy())

            # Thumbnail of what will be saved (bottom-right corner)
            thumb = cv2.resize(roi, (120, 120))
            th, tw = thumb.shape[:2]
            fh, fw = display.shape[:2]
            display[fh - th - 10 : fh - 10,
                    fw - tw - 10 : fw - 10] = thumb
            draw_rounded_rect(display,
                              (fw - tw - 12, fh - th - 12),
                              (fw - 8, fh - 8),
                              COL_WHITE, thickness=2, radius=6)
            put_text_shadow(display, "Preview",
                            (fw - tw - 12, fh - th - 16),
                            scale=0.42, color=COL_WHITE)

            cv2.imshow("Gesture Dataset Collector", display)

            # ── Key handling ─────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                logger.info("Quitting…")
                break

            elif key == ord(' '):           # Capture single image
                self.auto_capturing = False
                self.countdown_start = None
                self.capture_image(roi)

            elif key == ord('h'):           # Start countdown → auto-capture
                if not self.auto_capturing and self.countdown_start is None:
                    self.countdown_start  = time.time()
                    self.last_capture_time = time.time() + CAPTURE_COUNTDOWN
                    self._set_status("Get ready!", COL_ORANGE, CAPTURE_COUNTDOWN)
                elif self.auto_capturing:
                    self.auto_capturing  = False
                    self.countdown_start = None
                    self._set_status("Auto-capture stopped", COL_YELLOW, 1.5)

            elif key == ord('H'):           # Stop auto-capture (Shift+H)
                self.auto_capturing  = False
                self.countdown_start = None
                self._set_status("Auto-capture stopped", COL_YELLOW, 1.5)

            elif key == ord('n'):           # Next class
                self.auto_capturing  = False
                self.countdown_start = None
                self.current_idx = (self.current_idx + 1) % len(self.gestures)
                self._set_status(f"→ Class: {self.current_gesture} "
                                 f"[{self.current_count} imgs]", COL_CYAN, 1.5)

            elif key == ord('p'):           # Previous class
                self.auto_capturing  = False
                self.countdown_start = None
                self.current_idx = (self.current_idx - 1) % len(self.gestures)
                self._set_status(f"← Class: {self.current_gesture} "
                                 f"[{self.current_count} imgs]", COL_CYAN, 1.5)

            elif key == ord('d'):           # Delete last
                self.delete_last()

            elif key == ord('c'):           # Clear class
                self.clear_class()

            elif key == ord('s'):           # Stats
                self.print_stats()
                self._set_status("Stats printed to terminal", COL_WHITE, 1.5)

            elif key == ord('a'):           # Add new gesture
                self.release()
                name = input("\n  Enter new gesture name: ").strip().upper()
                if name and name not in self.gestures:
                    self.gestures.append(name)
                    self.current_idx = len(self.gestures) - 1
                    logger.info(f"Added gesture class: '{name}'")
                self.open_camera()

        # ── Cleanup & report ─────────────────────────────────────────────────
        self.release()
        self.print_stats()

        report_path = Path("dataset_session_report.json")
        report      = self.session.save(report_path)

        print("\n" + "═" * 55)
        print("  SESSION SUMMARY")
        print("═" * 55)
        print(f"  ✅  Total captured : {report['total_captured']}")
        print(f"  🗑   Total deleted  : {report['total_deleted']}")
        print(f"  📁  Saved to       : {self.dataset_dir}")
        print(f"  📋  Report         : {report_path}")
        print("═" * 55)
        print("\n  ▶  Next step: python train_cnn.py\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive gesture dataset collector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python collect_dataset.py                          # Use default settings
  python collect_dataset.py --camera 1               # External webcam
  python collect_dataset.py --gestures A B C D       # Collect only A,B,C,D
  python collect_dataset.py --start A                # Start from gesture A
  python collect_dataset.py --output datasets/mine   # Custom output folder
        """
    )
    parser.add_argument("--camera",   type=int, default=CAMERA_ID,
                        help=f"Camera device ID (default: {CAMERA_ID})")
    parser.add_argument("--gestures", nargs="+", default=None,
                        help="List of gesture classes to collect "
                             "(default: all from config)")
    parser.add_argument("--start",    type=str, default=None,
                        help="Gesture name to start at")
    parser.add_argument("--output",   type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--target",   type=int, default=TARGET_IMAGES,
                        help=f"Target images per class (default: {TARGET_IMAGES})")
    return parser.parse_args()


def main():
    args = parse_args()

    gestures    = args.gestures or GESTURE_NAMES
    dataset_dir = Path(args.output) if args.output else DATASETS_DIR

    global TARGET_IMAGES
    TARGET_IMAGES = args.target

    collector = DatasetCollector(
        gestures    = gestures,
        dataset_dir = dataset_dir,
        camera_id   = args.camera,
    )

    # Jump to starting gesture if specified
    if args.start:
        name = args.start.upper()
        if name in collector.gestures:
            collector.current_idx = collector.gestures.index(name)
        else:
            logger.warning(f"Gesture '{name}' not found — starting from beginning")

    print("\n" + "═" * 60)
    print("  🖐  GESTURE DATASET COLLECTOR")
    print("═" * 60)
    print(f"  Gestures   : {', '.join(gestures)}")
    print(f"  Target     : {TARGET_IMAGES} images/class")
    print(f"  Camera ID  : {args.camera}")
    print(f"  Output dir : {dataset_dir}")
    print("═" * 60)
    print("  CONTROLS:")
    print("  [SPACE]   Single capture")
    print("  [H]       Start/stop auto-capture burst")
    print("  [N] / [P] Next / Previous class")
    print("  [A]       Add custom gesture class")
    print("  [D]       Delete last image")
    print("  [C]       Clear all images for this class")
    print("  [S]       Print statistics")
    print("  [Q]       Quit & save session report")
    print("═" * 60 + "\n")

    collector.run()


if __name__ == "__main__":
    main()


