"""Hand Vocal FX — transforme ta voix selon les gestes de ta main.

Appuyez sur 'q' (ou Ctrl+C) pour quitter.
"""

import sys
import threading
import time

import cv2

import config as cfg
from audio import AudioEngine, print_devices
from hand import GestureTracker, HandTracker, draw_hand, fingers_down


def _open_camera():
    """Ouvre la caméra avec un backend compatible avec la plateforme.
    Windows : MSMF puis DirectShow. macOS : AVFoundation. Linux : défaut."""
    idx = cfg.CAMERA_INDEX
    if sys.platform.startswith("win"):
        backends = (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY)
    elif sys.platform == "darwin":
        backends = (cv2.CAP_AVFOUNDATION, cv2.CAP_ANY)
    else:
        backends = (cv2.CAP_ANY,)
    for backend in backends:
        try:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                return cap
            cap.release()
        except Exception:  # noqa: BLE001 - backend indisponible
            continue
    return cv2.VideoCapture(idx)


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.effect = "normal"
        self.gesture = ""
        self.fps = 0.0

    def set(self, effect=None, gesture=None, fps=None):
        with self.lock:
            if effect is not None:
                self.effect = effect
            if gesture is not None:
                self.gesture = gesture
            if fps is not None:
                self.fps = fps

    def snapshot(self):
        with self.lock:
            return self.effect, self.gesture, self.fps


def main():
    # affichage UTF-8 propre sur la console Windows
    if sys.platform.startswith("win"):
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                pass

    print("=" * 52)
    print("  Hand Vocal FX — contrôle vocal par gestes de la main")
    print("=" * 52)
    print_devices()
    print("Casque recommandé pour éviter le larsen.")
    print("Appuie sur 'q' dans la fenêtre pour quitter.\n")

    engine = AudioEngine()
    engine.start()

    tracker = HandTracker(cfg.MODEL_PATH)
    gesture = GestureTracker()
    state = SharedState()

    cap = _open_camera()
    if not cap.isOpened():
        print(f"ERREUR : impossible d'ouvrir la caméra {cfg.CAMERA_INDEX}.")
        engine.stop()
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.CAMERA_HEIGHT)

    last_effect = None
    frame_count = 0
    fps_t0 = time.time()
    empty_frames = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                empty_frames += 1
                if empty_frames == 1:
                    print("Aucune frame reçue de la caméra...")
                if empty_frames > 90:  # ~3s
                    print("\nPas de flux vidéo : vérifie que la caméra est "
                          "activée/connectée, et que CAMERA_INDEX dans "
                          "config.py est correct.")
                    break
                time.sleep(0.03)
                continue
            empty_frames = 0

            if cfg.MIRROR:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            hand = None
            frame_count += 1
            if frame_count % cfg.DETECT_EVERY_N_FRAMES == 0:
                hands = tracker.detect(frame)
                hand = hands[0] if hands else None

            if hand is not None:
                down = fingers_down(hand)
                eff = gesture.update(down)
                label = cfg.EFFECTS[eff]["label"]
                gesture_desc = ", ".join(down) if down else "main ouverte"
                draw_hand(frame, hand, w, h)
            else:
                eff = gesture.update(None)
                label = "Voix normale"
                gesture_desc = "main absente"

            # FPS
            now = time.time()
            if now - fps_t0 >= 1.0:
                state.set(fps=frame_count / max(now - fps_t0, 1e-6))
                frame_count = 0
                fps_t0 = now
            _, _, fps = state.snapshot()

            state.set(effect=eff, gesture=gesture_desc)
            if eff != last_effect:
                print(f"[effet] {label}  ({gesture_desc})")
                last_effect = eff
            engine.set_effect(eff)

            overlay = f"{label}  |  doigts baissés: {gesture_desc}  |  {fps:.0f} fps"
            cv2.putText(frame, overlay, (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2,
                        cv2.LINE_AA)
            cv2.imshow("Hand Vocal FX", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("\nArrêt.")


if __name__ == "__main__":
    main()