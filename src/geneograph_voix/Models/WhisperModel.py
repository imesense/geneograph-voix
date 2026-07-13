import inspect

import numpy as np

from faster_whisper import WhisperModel
from typing import Optional

from geneograph_voix.Models.Config import SAMPLERATE, _model_id_for_key
from geneograph_voix.Models.Devices import CPU_THREADS, DEVICE, NUM_WORKERS, _resolve_compute_type

# ===========================
# Whisper model loading + wrapper
# ===========================

model: Optional[WhisperModel] = None
COMPUTE_TYPE = None

def _pick_default_model_key() -> str:
    return "large-v3-turbo" if DEVICE == "cuda" else "base"

def _load_whisper_model(model_key: str):
    global model, COMPUTE_TYPE
    model_id = _model_id_for_key(model_key)
    chain, first = _resolve_compute_type(DEVICE)
    last_err = None
    for ct in chain:
        try:
            print(f"Loading Whisper model '{model_id}' on {DEVICE} with compute_type={ct} ...")
            model = WhisperModel(model_id, device=DEVICE, compute_type=ct)
            COMPUTE_TYPE = ct
            print(f"Whisper ready: compute_type={ct}")
            return
        except Exception as e:
            print(f"compute_type '{ct}' failed -> {e}")
            last_err = e
    if last_err is not None:
        raise last_err
    raise RuntimeError(f"No compute types available for device '{DEVICE}'")

def _warmup_model():
    if model is None:
        print("Warmup skipped: model not loaded yet")
        return
    try:
        dummy = np.zeros((SAMPLERATE // 2,), dtype=np.float32)
        list(fw_transcribe(dummy, beam_size=1, temperature=0.0, without_timestamps=True))
    except Exception as e:
        print("Warmup failed (non-fatal):", e)

def fw_transcribe(audio, **kwargs):
    if model is None:
        raise RuntimeError("Whisper model not loaded. Call _load_whisper_model first.")
    if CPU_THREADS is not None:
        kwargs.setdefault("cpu_threads", CPU_THREADS)
    if NUM_WORKERS is not None:
        kwargs.setdefault("num_workers", NUM_WORKERS)
    params = inspect.signature(model.transcribe).parameters
    safe_kwargs = {k: v for k, v in kwargs.items() if k in params}
    return model.transcribe(audio, **safe_kwargs)
