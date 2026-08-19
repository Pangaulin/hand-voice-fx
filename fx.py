"""DSP temps réel (numpy) : pitch shift (phase vocoder), autotune, robot, réverb.

Tous les traitements se font sur des blocs float32 mono. Chaque classe est
streaming : on l'appelle avec un bloc d'entrée et elle renvoie la sortie
disponible (même longueur en régime permanent).
"""

import numpy as np

import config as cfg


# ---------------------------------------------------------------------------
# Ligne à retard simple (pour aligner sec/humide)
# ---------------------------------------------------------------------------
class Delay:
    def __init__(self, n):
        self.n = int(n)
        self.buf = np.zeros(self.n, np.float32)

    def reset(self):
        self.buf.fill(0.0)

    def process(self, x):
        n = self.n
        if n == 0 or x.size == 0:
            return x.copy()
        out = np.concatenate([self.buf, x])
        self.buf = out[-n:].copy()
        return out[: x.size]


# ---------------------------------------------------------------------------
# Pitch shifter temps réel — phase vocoder verrouillé (Laroche & Dolson 1999)
#   On détecte les pics spectraux, on déplace chaque lobe d'un décalage entier
#   de bins, et on verrouille la phase de tous les bins d'un lobe sur celle de
#   leur pic (identity phase locking) : les relations de phase sont préservées
#   et l'overlap-add reste cohérent -> pitch exact, durée conservée, pas de
#   smearing/phasiness, énergie préservée (avec compensation du "scalloping").
# ---------------------------------------------------------------------------
class PitchShifter:
    def __init__(self, sr=cfg.SAMPLE_RATE, n_fft=cfg.N_FFT, hop=cfg.HOP,
                 semitones=0.0):
        self.sr = sr
        self.n = n_fft
        self.ha = hop
        # fenêtre de Hann périodique (celle du modèle de scalloping ci-dessous)
        self.win = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft)
                    ).astype(np.float32)
        self.nbins = n_fft // 2 + 1
        self.half = self.nbins - 1
        self.bins = np.arange(self.nbins)
        self.freq_per_bin = 2.0 * np.pi / n_fft   # rad/sample par bin
        self._inbuf = np.zeros(0, np.float32)
        self._out = np.zeros(n_fft, np.float32)
        self._out_index = 0
        self._syn = np.zeros(self.nbins)          # phase de synthèse (par bin)
        self._prev_phase = np.zeros(self.nbins)
        self._first = True
        self._norm = self._compute_norm(hop)
        self.set_semitones(semitones)

    def set_semitones(self, semitones):
        self.semitones = float(semitones)
        self.ratio = 2.0 ** (self.semitones / 12.0)

    def _compute_norm(self, hs):
        # Somme de w^2 sur les fenêtres chevauchées à l'espacement hs (périodique)
        n, win = self.n, self.win
        w2 = win * win
        norm = np.zeros(hs, np.float32)
        for t in range(hs):
            s = 0.0
            kmin = (t - n + 1) // hs
            for k in range(kmin, kmin + n // hs + 2):
                pos = t - k * hs
                if 0 <= pos < n:
                    s += w2[pos]
            norm[t] = s
        norm[norm < 1e-3] = 1.0
        return norm

    def reset(self):
        self._inbuf = np.zeros(0, np.float32)
        self._out.fill(0.0)
        self._out_index = 0
        self._syn.fill(0.0)
        self._prev_phase.fill(0.0)
        self._first = True

    def process(self, x):
        x = np.asarray(x, dtype=np.float32)
        self._inbuf = np.concatenate([self._inbuf, x])
        chunks = []
        while self._inbuf.size >= self.n:
            frame = self._inbuf[: self.n]
            self._inbuf = self._inbuf[self.ha:]
            chunks.append(self._process_frame(frame))
        if not chunks:
            return np.zeros(0, np.float32)
        return np.concatenate(chunks)

    @staticmethod
    def _wrap(p):
        return p - np.round(p / (2.0 * np.pi)) * (2.0 * np.pi)

    def _nearest_peak(self, peaks):
        n = peaks.size
        if n <= 1:
            return np.zeros(self.nbins, dtype=np.int64)
        idx = np.searchsorted(peaks, self.bins)
        idx = np.clip(idx, 1, n - 1)
        lo, hi = idx - 1, idx
        near = np.where(np.abs(peaks[lo] - self.bins) <= np.abs(peaks[hi] - self.bins),
                        lo, hi)
        return near

    def _lobe_gain(self, dw):
        # amplitude moyenne de l'overlap-add d'une partielle dont la fréquence
        # intra-trame (grille) et inter-trame (réelle) diffèrent de dw rad/éch.
        dw = np.asarray(dw, dtype=np.float64)
        b = 2.0 * np.pi / self.n

        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.sin(dw / 2.0)
            d_dw = np.where(np.abs(s) < 1e-12, self.n,
                            np.sin(self.n * dw / 2.0) / s)
            s = np.sin((dw - b) / 2.0)
            d_low = np.where(np.abs(s) < 1e-12, self.n,
                             np.sin(self.n * (dw - b) / 2.0) / s)
            s = np.sin((dw + b) / 2.0)
            d_hi = np.where(np.abs(s) < 1e-12, self.n,
                            np.sin(self.n * (dw + b) / 2.0) / s)

        g = np.abs(0.5 * d_dw + 0.25 * d_low + 0.25 * d_hi) / (0.5 * self.n)
        return np.where(np.abs(dw) < 1e-12, 1.0, g)

    def _process_frame(self, frame):
        X = np.fft.rfft(frame * self.win)
        mag = np.abs(X).astype(np.float64)
        phase = np.angle(X).astype(np.float64)
        half = self.half

        # --- détection des pics (max locaux au-dessus d'un plancher) ---
        maxm = float(mag.max())
        floor = max(1e-8, maxm * 0.005)
        m = mag[1:-1]
        peaks = np.where((m >= mag[:-2]) & (m > mag[2:]) & (m >= floor))[0] + 1
        npeaks = peaks.size

        newMag = np.zeros(self.nbins)
        newPhase = np.zeros(self.nbins)
        peak_dest = np.zeros(0, dtype=np.int64)
        peak_syn = np.zeros(0)
        boost = np.zeros(0)

        if npeaks:
            # fréquence instantanée de chaque pic, puis destination du lobe
            if self._first:
                true_freq = peaks * self.freq_per_bin
            else:
                dp = self._wrap(phase[peaks] - self._prev_phase[peaks]
                                - peaks * self.freq_per_bin * self.ha)
                true_freq = peaks * self.freq_per_bin + dp / self.ha
            shifted = true_freq * self.ratio
            dest_bin = peaks + np.round(
                (shifted - true_freq) / self.freq_per_bin).astype(np.int64)
            ok = (dest_bin >= 0) & (dest_bin <= half)

            if self._first:
                new_syn = phase[peaks].copy()
            else:
                db = np.clip(dest_bin, 0, half)
                new_syn = self._wrap(self._syn[db] + shifted * self.ha)
            self._syn[dest_bin[ok]] = new_syn[ok]
            peak_dest = np.full(npeaks, -1, dtype=np.int64)
            peak_dest[ok] = dest_bin[ok]
            peak_syn = new_syn.copy()

            dw = shifted - (true_freq + (dest_bin - peaks) * self.freq_per_bin)
            r = self._lobe_gain(dw)
            boost = np.where(r > 1e-12, 1.0 / (r * r), 1.0)

        self._first = False
        self._prev_phase = phase.copy()

        # --- repli rigide : chaque bin suit son pic le plus proche ---
        if npeaks:
            pi = self._nearest_peak(peaks)
            pk = peaks[pi]
            pd = peak_dest[pi]
            valid = pd >= 0
            dest = pd + (self.bins - pk)
            inband = valid & (dest >= 0) & (dest <= half)

            e = mag * mag
            eIn = float(e[valid].sum())
            eOut = float(e[inband].sum())

            if inband.any():
                np.add.at(newMag, dest[inband],
                          e[inband] * boost[pi[inband]])
                # phase : le contributeur le plus fort du bin gagne
                ph_dest = peak_syn[pi] + (phase - phase[pk])
                ib_dest = dest[inband]
                ib_ph = ph_dest[inband]
                ib_mag = mag[inband]
                order = np.lexsort((ib_mag, ib_dest))
                ds = ib_dest[order]
                ps = ib_ph[order]
                group_start = np.flatnonzero(np.r_[True, ds[1:] != ds[:-1]])
                ends = np.r_[group_start[1:] - 1, order.size - 1]
                newPhase[ds[ends]] = ps[ends]

            # compensation : énergie du repli (y compris pertes au-delà de Nyquist)
            if eOut > 1e-24 and eIn > 1e-24:
                newMag = np.sqrt(newMag) * np.sqrt(eIn / eOut)

        # --- synthèse + overlap-add ---
        spec = newMag * np.exp(1j * newPhase)
        f = np.fft.irfft(spec, n=self.n).astype(np.float32) * self.win
        self._out += f
        emit = self._out[: self.ha].copy()
        idx = (np.arange(self.ha, dtype=np.int32) + self._out_index) % self.ha
        emit /= self._norm[idx]
        emit *= cfg.PITCH_MAKEUP_GAIN
        emit = np.clip(emit, -1.0, 1.0)
        self._out[: self.n - self.ha] = self._out[self.ha:]
        self._out[self.n - self.ha:] = 0.0
        self._out_index += self.ha
        return emit


# ---------------------------------------------------------------------------
# Détection de pitch (algorithme YIN)
# ---------------------------------------------------------------------------
def detect_pitch(x, sr):
    x = np.asarray(x, dtype=np.float32)
    if x.size < 128:
        return 0.0, 0.0
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x * x)))
    if rms < 1e-4:
        return 0.0, 0.0

    fmin, fmax = 70.0, 1200.0
    minlag = max(1, int(sr / fmax))
    maxlag = min(int(sr / fmin), x.size - 1)
    if maxlag <= minlag:
        return 0.0, 0.0

    r = np.correlate(x, x, mode="full")
    nlen = x.size
    r0 = r[nlen - 1]
    if r0 <= 1e-6:
        return 0.0, 0.0

    # fonction de différence : d(tau) = 2*(r0 - r(tau))
    d = 2.0 * (r0 - r[nlen - 1: nlen + maxlag])
    d = np.maximum(d, 0.0)

    # CMNDF (cumulative mean normalized difference) : normalise par la moyenne
    d1 = d[1:]
    cm = np.cumsum(d1) / np.arange(1.0, d1.size + 1.0)
    dcm = d1 / np.maximum(cm, 1e-9)

    taus = np.arange(1, maxlag + 1)
    mask = (taus >= minlag) & (taus <= maxlag)
    dcm = dcm[mask]
    taus = taus[mask]
    if dcm.size == 0:
        return 0.0, 0.0

    below = np.where(dcm < cfg.YIN_THRESHOLD)[0]
    if below.size > 0:
        j = below[0]
        window = dcm[j: min(j + 8, dcm.size)]
        k = j + int(np.argmin(window))
    else:
        k = int(np.argmin(dcm))
    lag = int(taus[k])
    clarity = 1.0 - float(dcm[k])
    if clarity <= 0.0:
        return 0.0, 0.0

    # interpolation parabolique du minimum
    if 0 < k < dcm.size - 1:
        prev, cur, nxt = float(dcm[k - 1]), float(dcm[k]), float(dcm[k + 1])
        denom = prev - 2.0 * cur + nxt
        if denom > 1e-9:
            lag += (prev - nxt) / (2.0 * denom)
    if lag <= 0:
        return 0.0, 0.0
    f0 = sr / lag
    return f0, clarity


# ---------------------------------------------------------------------------
# Autotune
# ---------------------------------------------------------------------------
class Autotune:
    def __init__(self, sr=cfg.SAMPLE_RATE, n=cfg.PITCH_LATENCY_N):
        self.sr = sr
        self.n = n
        self.buf = np.zeros(n, np.float32)
        self.shifter = PitchShifter(sr=sr, semitones=0.0)
        self.dry = Delay(cfg.N_FFT)
        self._smooth = 1.0
        self._g = 0.0
        self._alpha_a = 0.35
        self._alpha_r = 0.10
        self._scale = np.asarray(cfg.AUTOTUNE_SCALE_SEMITONES, dtype=np.float64)

    def reset(self):
        self.buf.fill(0.0)
        self.shifter.reset()
        self.dry.reset()
        self._smooth = 1.0
        self._g = 0.0

    def _target_semitones(self, f0):
        midi = 69.0 + 12.0 * np.log2(f0 / 440.0)
        base = np.floor(midi / 12.0) * 12.0
        cand = []
        for octave in (-1, 0, 1):
            cand.extend(base + octave * 12.0 + self._scale)
        cand = np.asarray(cand, dtype=np.float64)
        target = float(cand[int(np.argmin(np.abs(cand - midi)))])
        shift = target - midi
        return float(np.clip(shift, -cfg.AUTOTUNE_MAX_SHIFT_ST,
                             cfg.AUTOTUNE_MAX_SHIFT_ST))

    def process(self, x):
        x = np.asarray(x, dtype=np.float32)
        # fenêtre glissante pour la détection
        self.buf = np.concatenate([self.buf, x])[-self.n:]

        f0, clarity = detect_pitch(self.buf, self.sr)
        voiced = f0 > 0.0 and clarity >= cfg.AUTOTUNE_MIN_CLARITY

        if voiced:
            shift_st = self._target_semitones(f0)
            target_ratio = 2.0 ** (shift_st / 12.0)
            a = self._alpha_a if target_ratio > self._smooth else self._alpha_r
        else:
            target_ratio = 1.0
            a = self._alpha_r
        self._smooth += a * (target_ratio - self._smooth)

        self.shifter.set_semitones(12.0 * np.log2(max(self._smooth, 1e-6)))
        wet = self.shifter.process(x)

        # fondu sec/humide (aligné en latence) + force de l'autotune
        dry = self.dry.process(x)
        target_g = 1.0 if voiced else 0.0
        self._g += 0.2 * (target_g - self._g)
        w = self._g * cfg.AUTOTUNE_WEIGHT
        m = min(dry.size, wet.size)
        if m == 0:
            return dry
        out = np.empty(dry.size, dtype=np.float32)
        out[:m] = (1.0 - w) * dry[:m] + w * wet[:m]
        if wet.size > m:
            out[m:] = (1.0 - w) * dry[m:] + w * wet[m:]
        else:
            out[m:] = dry[m:]
        return out


# ---------------------------------------------------------------------------
# Robot : ring modulation (carré) + saturation
# ---------------------------------------------------------------------------
class Robot:
    def __init__(self, sr=cfg.SAMPLE_RATE, freq=cfg.ROBOT_FREQ,
                 drive=cfg.ROBOT_DRIVE):
        self.sr = sr
        self.freq = freq
        self.drive = drive
        self.t0 = 0.0

    def reset(self):
        self.t0 = 0.0

    def process(self, x):
        x = np.asarray(x, dtype=np.float32)
        t = self.t0 + np.arange(x.size, dtype=np.float32)
        self.t0 += x.size
        mod = 0.5 + 0.5 * np.sign(np.sin(2.0 * np.pi * self.freq * t / self.sr))
        y = x * mod
        y = np.tanh(self.drive * y) / np.tanh(self.drive)
        return y.astype(np.float32)


# ---------------------------------------------------------------------------
# Réverb : écho multi-taps décroissant (façon grotte)
# ---------------------------------------------------------------------------
class Reverb:
    def __init__(self, sr=cfg.SAMPLE_RATE, mix=cfg.REVERB_MIX):
        self.sr = sr
        self.mix = mix
        taps = [(0.021, 0.80), (0.033, 0.62), (0.047, 0.42), (0.061, 0.25)]
        self.delays = [Delay(int(d * sr)) for d, _ in taps]
        self.gains = [g for _, g in taps]

    def reset(self):
        for dl in self.delays:
            dl.reset()

    def process(self, x):
        x = np.asarray(x, dtype=np.float32)
        wet = np.zeros_like(x)
        for dl, g in zip(self.delays, self.gains):
            wet += g * dl.process(x)
        return ((1.0 - self.mix) * x + self.mix * wet).astype(np.float32)


# ---------------------------------------------------------------------------
# Chaîne d'effets pilotée par le geste
# ---------------------------------------------------------------------------
class VoiceFx:
    def __init__(self, sr=cfg.SAMPLE_RATE):
        self.sr = sr
        self._shifters = {}
        self._autotune = Autotune(sr=sr)
        self._robot = Robot(sr=sr)
        self._reverb = Reverb(sr=sr)
        self._dry_delay = Delay(cfg.N_FFT)
        self._current = "normal"
        self._crossfade = 0

    def _shifter_for(self, name):
        if name not in self._shifters:
            self._shifters[name] = PitchShifter(sr=self.sr)
        return self._shifters[name]

    def set_effect(self, name):
        if name == self._current:
            return
        self._current = name
        self._crossfade = cfg.CROSSFADE_SAMPLES
        self._dry_delay.reset()
        if name == "autotune" or name == "robot_autotune" or name == "autotune_aigu":
            self._autotune.reset()
        if name in self._shifters:
            self._shifters[name].reset()

    def process(self, x):
        spec = cfg.EFFECTS.get(self._current, {})
        dry = np.asarray(x, dtype=np.float32)
        y = dry

        if spec.get("pitch"):
            sh = self._shifter_for(self._current)
            sh.set_semitones(spec["pitch"])
            y = sh.process(dry)
        elif spec.get("autotune"):
            y = self._autotune.process(dry)

        if spec.get("reverb"):
            y = self._reverb.process(y)
        if spec.get("robot"):
            y = self._robot.process(y)

        if spec.get("dry_mix"):
            dry_aligned = self._dry_delay.process(dry)
            out = np.empty(dry_aligned.size, dtype=np.float32)
            m = min(dry_aligned.size, y.size)
            if m > 0:
                out[:m] = (spec["dry_mix"] * dry_aligned[:m]
                           + (1.0 - spec["dry_mix"]) * y[:m])
            if y.size > m:
                out[m:] = y[m:]
            else:
                out[m:] = dry_aligned[m:]
            y = out

        # fondu anti-clic lors d'un changement d'effet
        if self._crossfade > 0 and y.size > 0:
            n = min(self._crossfade, y.size)
            ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
            y = y.copy()
            y[:n] = (1.0 - ramp) * dry[:n] + ramp * y[:n]
            self._crossfade -= n

        return y