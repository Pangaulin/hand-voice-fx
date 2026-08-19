# Hand Vocal FX

Transforme ta voix en temps réel selon les gestes de ta main (caméra + MediaPipe) :
1 doigt = voix aiguë, 2 = grave, 3 = double, 4 = fantôme, poing = Vador,
main ouverte = Robot, index + majeur = Autotune, paume ouverte bougeante = normal.

Fonctionne sur **Linux**, **Windows** et **macOS**.

## Prérequis

- Python **3.9 ou plus récent** (testé avec 3.12+).
- Une webcam (ou un flux `/dev/video0` sur Linux) et un micro.
- **Casque conseillé** pour éviter le larsen.

## Installation (les 3 OS)

1. **Télécharger le modèle** (une seule fois) — il est déjà fourni dans `models/`.
   S'il est absent :
   ```
   curl -L -o models/hand_landmarker.task \
     https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
   ```

2. **Créer l'environnement virtuel** :
   ```bash
   python -m venv .venv
   ```

3. **Installer les dépendances** :
   ```bash
   # Linux / macOS
   .venv/bin/python -m pip install -r requirements.txt

   # Windows (PowerShell ou cmd)
   .venv\Scripts\python -m pip install -r requirements.txt
   ```

## Lancer

| OS       | Commande                              |
|----------|---------------------------------------|
| Linux    | `./run.sh`                            |
| macOS    | `./run.sh` (ou `bash run.sh`)         |
| Windows  | `run.bat` (double-clic ou `run.bat`)  |

Touches : `q` pour quitter dans la fenêtre vidéo, `Ctrl+C` dans le terminal.

## Notes par plateforme

### Linux
- Python : `sudo pacman -S python python-pip` (Arch), `sudo apt install python3 python3-venv` (Debian/Ubuntu).
- Le son se fait via **PulseAudio/PipeWire** (automatique).
- Avec Droidcam (téléphone en webcam) : lance l'appli, puis le client `droidcam` ;
  `CAMERA_INDEX=0` pointe alors sur `/dev/video0`.

### Windows
- Python : https://www.python.org/downloads/ (coche **"Add Python to PATH"**).
- La caméra et le micro demandent leur autorisation dans **Paramètres > Confidentialité**.
- Backend caméra : MSMF puis DirectShow automatiques.

### macOS
- Python : `brew install python` ou https://www.python.org/downloads/.
- Autoriser **Micro** et **Caméra** dans **Réglages Système > Confidentialité et sécurité**.
- Backend caméra : AVFoundation automatique.
- `run.sh` : `chmod +x run.sh` si besoin.

## Configuration

Tout est dans `config.py` : index de la caméra, gains d'entrée/sortie
(`INPUT_GAIN`, `OUTPUT_GAIN`), taille FFT, latence, mapping gestes→effets.

## Dépannage

- **Pas de son** : vérifie dans `config.py` que `DEVICE_IN`/`DEVICE_OUT`
  sont `None` (choix auto) ou renseigne l'index affiché au démarrage.
- **Son faible** : augmente `INPUT_GAIN` (micro faible) et/ou `OUTPUT_GAIN`.
- **Pas de vidéo** : teste la webcam avec `cv2.VideoCapture(0)` dans Python ;
  sinon change `CAMERA_INDEX` dans `config.py`.
- **Larsen** : mets un casque.