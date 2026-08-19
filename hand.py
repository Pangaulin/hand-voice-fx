"""Détection de main MediaPipe + comptage des doigts baissés + mapping geste->effet."""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import config as cfg

# Connexions du squelette de la main (indices de landmarks 0..20)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# pouce : tip=4, IP=3, index MCP=5
TIPS = {"pouce": 4, "index": 8, "majeur": 12, "annulaire": 16, "auriculaire": 20}
PIPS = {"index": 6, "majeur": 10, "annulaire": 14, "auriculaire": 18}


class HandTracker:
    def __init__(self, model_path=cfg.MODEL_PATH, num_hands=1):
        base = mp_python.BaseOptions(model_asset_path=model_path)
        opts = vision.HandLandmarkerOptions(
            base_options=base,
            num_hands=num_hands,
            min_hand_detection_confidence=cfg.MIN_HAND_CONFIDENCE,
            min_hand_presence_confidence=cfg.MIN_HAND_CONFIDENCE,
            min_tracking_confidence=cfg.MIN_HAND_CONFIDENCE,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(opts)

    def detect(self, frame_bgr):
        """Renvoie la liste des mains détectées (chaque main = 21 points (x,y,z))."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.landmarker.detect(mp_img)
        hands = []
        for lm_list in res.hand_landmarks:
            pts = [(float(lm.x), float(lm.y), float(lm.z)) for lm in lm_list]
            hands.append(pts)
        return hands


def fingers_down(hand):
    """Liste des doigts baissés pour une main (21 points normalisés)."""
    down = []
    for name, tip_i in TIPS.items():
        if name == "pouce":
            continue
        tip_y = hand[tip_i][1]
        pip_y = hand[PIPS[name]][1]
        if tip_y > pip_y + cfg.FINGER_THRESHOLD:
            down.append(name)
    # pouce baissé si sa pointe passe sous son IP (replié vers la paume)
    if hand[4][1] > hand[3][1] + cfg.FINGER_THRESHOLD:
        down.append("pouce")
    return down


def resolve_effect(key):
    if key in cfg.GESTURES:
        return cfg.GESTURES[key]
    for f in cfg.FALLBACK_PRIORITY:
        if f in key:
            return cfg.GESTURES.get((f,), "normal")
    return "normal"


def canonical_key(down_fingers):
    return tuple(f for f in cfg.FINGER_ORDER if f in down_fingers)


class GestureTracker:
    def __init__(self):
        self._stable = None
        self._count = 0
        self.current = "normal"

    def update(self, down_fingers):
        """down_fingers : None (pas de main) ou liste de doigts baissés."""
        if down_fingers is None:
            self._stable = None
            self._count = 0
            self.current = resolve_effect(())
            return self.current
        key = canonical_key(down_fingers)
        if key == self._stable:
            self._count += 1
        else:
            self._stable = key
            self._count = 1
        if self._count >= cfg.STABILITY_FRAMES:
            self.current = resolve_effect(key)
        return self.current


def draw_hand(frame, hand, w, h):
    for a, b in HAND_CONNECTIONS:
        p1 = (int(hand[a][0] * w), int(hand[a][1] * h))
        p2 = (int(hand[b][0] * w), int(hand[b][1] * h))
        cv2.line(frame, p1, p2, (0, 255, 0), 2)
    for p in hand:
        cv2.circle(frame, (int(p[0] * w), int(p[1] * h)), 4, (0, 0, 255), -1)