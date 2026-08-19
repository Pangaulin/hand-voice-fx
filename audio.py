"""Flux audio temps réel : micro -> chaîne d'effets -> sortie (casque)."""

import threading

import numpy as np
import sounddevice as sd

import config as cfg
from fx import VoiceFx


class AudioEngine:
    def __init__(self, sr=cfg.SAMPLE_RATE, block=cfg.BLOCK_SIZE):
        self.sr = sr
        self.block = block
        self.fx = VoiceFx(sr=sr)
        self.stream = None
        self._lock = threading.Lock()
        self._effect = "normal"
        self._residual = np.zeros(0, np.float32)

    def set_effect(self, name):
        with self._lock:
            if name != self._effect:
                self._effect = name

    def _callback(self, indata, outdata, frames, time_info, status):
        if status:
            print("audio:", status, file=open("/dev/null", "w"))
        with self._lock:
            eff = self._effect
        self.fx.set_effect(eff)

        x = np.asarray(indata[:, 0], dtype=np.float32) * cfg.INPUT_GAIN
        y = self.fx.process(x)

        # tampon résiduel : la chaîne d'effets peut sortir moins/moins de
        # frames que la taille du bloc (latence du phase vocoder) -> on ne
        # perd jamais d'échantillons, on décale.
        buf = np.concatenate([self._residual, y])
        if buf.size >= frames:
            out = buf[:frames]
            self._residual = buf[frames:]
        else:
            out = np.zeros(frames, dtype=np.float32)
            out[: buf.size] = buf
            self._residual = np.zeros(0, np.float32)

        # limiteur doux (tanh) : plus fort sans écrêtage dur
        out = np.tanh(out * cfg.OUTPUT_GAIN).astype(np.float32)
        outdata[:] = out[:, None]

    def start(self):
        self.stream = sd.Stream(
            samplerate=self.sr,
            blocksize=self.block,
            channels=1,
            dtype="float32",
            device=(cfg.DEVICE_IN, cfg.DEVICE_OUT),
            latency="low",
            callback=self._callback,
        )
        self.stream.start()

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None


def list_devices():
    return sd.query_devices()


def print_devices():
    try:
        print(sd.query_devices())
    except Exception as e:  # noqa: BLE001
        print("Impossible de lister les périphériques audio:", e)