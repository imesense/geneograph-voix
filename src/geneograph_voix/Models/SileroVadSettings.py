import torch

import numpy as np

from typing import Optional

from geneograph_voix.Models.Config import SAMPLERATE
from geneograph_voix.Models.Devices import DEVICE
from geneograph_voix.Models.ModelCoefficients import (
    VAD_STRICTNESS,
    _silero_params_for
)

# ===========================
# Silero VAD only
# ===========================

HAS_SILERO = False
SILERO_DEVICE = DEVICE

def _load_silero_vad():
    global _silero_model, _get_speech_ts
    try:
        _silero_model, _silero_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            trust_repo=True,
            force_reload=False
        )
        (_get_speech_ts, *_rest) = _silero_utils
        _silero_model.to(SILERO_DEVICE)
        _silero_model.eval()
        if SILERO_DEVICE == "cpu":
            try:
                torch.set_num_threads(1)
            except Exception:
                pass
        HAS_SILERO = True
        print(f"Silero VAD ready on {SILERO_DEVICE}")
    except Exception as e:
        print("Silero VAD not available:", e)
        HAS_SILERO = False

def _silero_vad_trim(buf_f32: np.ndarray, sr: int = SAMPLERATE) -> Optional[np.ndarray]:
    if not HAS_SILERO or buf_f32 is None or getattr(buf_f32, "size", 0) == 0:
        return buf_f32
    p = _silero_params_for(VAD_STRICTNESS)
    wav = torch.from_numpy(buf_f32).float()
    ts = _get_speech_ts(
        wav, _silero_model,
        sampling_rate=sr,
        threshold=p["th"],
        min_speech_duration_ms=p["min_speech"],
        min_silence_duration_ms=p["min_silence"],
        speech_pad_ms=p["pad"],
    )
    if not ts:
        return None
    start = ts[0]["start"]; end = ts[-1]["end"]
    trimmed = buf_f32[start:end]
    if len(trimmed) < int(sr * 0.20):
        return None
    return trimmed
