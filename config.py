"""Configuration : mapping gestes -> effets et paramètres audio/vidéo.

Modifie librement ce fichier pour personnaliser le comportement.
"""

import os

# ---------------------------------------------------------------------------
# Chemin du modèle MediaPipe
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "hand_landmarker.task")

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
SAMPLE_RATE = 48000
BLOCK_SIZE = 1024                 # taille des blocs traités (en échantillons)
N_FFT = 4096                      # taille de fenêtre du phase vocoder
HOP = N_FFT // 8                  # pas d'analyse (512)
PITCH_LATENCY_N = 2048            # fenêtre de détection de pitch (autotune)

DEVICE_OUT = None                 # None = sortie par défaut
DEVICE_IN = None                  # None = micro par défaut
CHANNELS = 1
INPUT_GAIN = 3.0                 # amplification du micro (compense un micro faible)
OUTPUT_GAIN = 2.0                # gain de sortie global (casque, limiteur doux tanh)

# ---------------------------------------------------------------------------
# Vidéo
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0                  # 0 = webcam par défaut (Linux: /dev/video0)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
MIRROR = True                     # True si tu veux un retour miroir
MIN_HAND_CONFIDENCE = 0.5
STABILITY_FRAMES = 4              # nb de frames stables avant de changer d'effet
DETECT_EVERY_N_FRAMES = 1         # 1 = détection à chaque frame
FINGER_THRESHOLD = 0.02           # seuil (coordonnées normalisées) pour "doigt baissé"

# ---------------------------------------------------------------------------
# Mapping doigts
# ---------------------------------------------------------------------------
# Ordre canonique des doigts (utilisé pour normaliser les combinaisons)
FINGER_ORDER = ["pouce", "index", "majeur", "annulaire", "auriculaire"]

# ---------------------------------------------------------------------------
# Effets disponibles
# ---------------------------------------------------------------------------
EFFECTS = {
    "normal":         {"label": "Voix normale"},
    "aigu":           {"label": "Voix aiguë",       "pitch": 5},
    "grave":          {"label": "Voix grave",       "pitch": -5},
    "autotune":       {"label": "Autotune",         "autotune": True},
    "robot":          {"label": "Robot",            "robot": True},
    "fantome":        {"label": "Fantôme",          "reverb": True, "pitch": 2},
    "double":         {"label": "Double voix",      "pitch": 12, "dry_mix": 0.5},
    "autotune_aigu":  {"label": "Autotune + aigu",  "autotune": True, "pitch": 5},
    "grave_reverb":   {"label": "Grave + réverb",   "pitch": -5, "reverb": True},
    "robot_autotune": {"label": "Robot + autotune", "robot": True, "autotune": True},
    "vador":          {"label": "Dark Vador",       "pitch": -7, "reverb": True},
}

# ---------------------------------------------------------------------------
# Mapping gestes -> effets
#   Clé = tuple de doigts BAISSÉS (ordre canonique), () = aucun doigt baissé
# ---------------------------------------------------------------------------
GESTURES = {
    ():                                       "normal",
    ("pouce",):                               "aigu",
    ("index",):                               "grave",
    ("majeur",):                              "autotune",
    ("annulaire",):                           "robot",
    ("auriculaire",):                         "fantome",
    ("pouce", "index"):                       "double",
    ("pouce", "majeur"):                      "autotune_aigu",
    ("index", "majeur"):                      "grave_reverb",
    ("pouce", "index", "majeur"):             "robot_autotune",
    ("pouce", "index", "majeur",
     "annulaire", "auriculaire"):             "vador",
}

# Repli si une combinaison n'est pas dans GESTURES :
# on prend l'effet du doigt le plus "important" parmi les doigts baissés.
# (ordre du plus important au moins important)
FALLBACK_PRIORITY = ["auriculaire", "annulaire", "majeur", "index", "pouce"]

# ---------------------------------------------------------------------------
# Paramètres DSP
# ---------------------------------------------------------------------------
PITCH_MIX = 1.0                   # mix sec/humide du pitch shifter (0..1)
PITCH_MAKEUP_GAIN = 1.0           # le shifter préserve l'énergie ; gain neutre
AUTOTUNE_WEIGHT = 0.85           # force de l'autotune (0..1)
AUTOTUNE_MAX_SHIFT_ST = 6        # correction max (en demi-tons)
AUTOTUNE_MIN_CLARITY = 0.35      # confiance pitch (voix) min pour corriger
YIN_THRESHOLD = 0.15             # seuil de plongée CMNDF pour la détection de pitch
ROBOT_FREQ = 55.0                 # fréquence de la ring modulation (Hz)
ROBOT_DRIVE = 2.5                 # saturation (tanh)
REVERB_MIX = 0.35                 # mix sec/réverb (0..1)
REVERB_DECAY = 0.72               # feedback des combs
CROSSFADE_SAMPLES = 512           # fondu à chaque changement d'effet (anti-clic)

# gamme utilisée par l'autotune (demi-tons par rapport à C) - chromatique par défaut
AUTOTUNE_SCALE_SEMITONES = list(range(12))