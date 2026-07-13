import numpy as np

from geneograph_voix.Helpers.Audio import (
    is_silence,
    rms_db
)
from geneograph_voix.Helpers.Language import (
    _effective_language,
    looks_like_outro,
    strip_trailing_dot
)
from geneograph_voix.Models.ModelCoefficients import (
    _energy_gate_db,
    _no_speech_thresholds
)
from geneograph_voix.Models.SileroVad import (
    HAS_SILERO,
    _silero_vad_trim
)
from geneograph_voix.Models.WhisperModel import (
    fw_transcribe
)

# ===========================
# Decoding functions
# ===========================

def transcribe_buffer(buffer):
    if buffer.size == 0:
        return ""
    if rms_db(buffer.flatten()) < (_energy_gate_db() + 2.0):
        return ""
    prev_nst, _, _, cr_prev, _ = _no_speech_thresholds()
    segments, _ = fw_transcribe(
        buffer,
        language=_effective_language(),
        beam_size=1,
        temperature=0.0,
        without_timestamps=True,
        condition_on_previous_text=False,
        no_speech_threshold=prev_nst,
        compression_ratio_threshold=cr_prev,
    )
    return " ".join((seg.text or "").strip() for seg in segments).strip()

def transcribe_buffer_commit(buffer):
    if buffer is None or buffer.size == 0:
        return ""
    mono = buffer[:, 0] if getattr(buffer, "ndim", 0) > 1 else buffer
    mono = np.asarray(mono, dtype=np.float32, order="C")
    if rms_db(mono) < _energy_gate_db():
        return ""
    trimmed = mono
    try:
        if HAS_SILERO:
            t = _silero_vad_trim(mono)
            if t is not None:
                trimmed = t
            else:
                if not is_silence(mono):
                    trimmed = mono
                else:
                    return ""
    except Exception:
        pass
    if trimmed.shape[0] > 1:
        x = trimmed.copy()
        x[1:] = x[1:] - 0.97 * x[:-1]
        trimmed = x
    samples = trimmed.flatten()
    prev_nst, commit1_nst, commit2_nst, _, cr_commit = _no_speech_thresholds()
    segments, info = fw_transcribe(
        samples,
        language=_effective_language(),
        beam_size=1,
        temperature=0.0,
        without_timestamps=True,
        condition_on_previous_text=False,
        no_speech_threshold=commit1_nst,
        compression_ratio_threshold=cr_commit,
    )
    segs = list(segments)
    full_text = " ".join((s.text or "").strip() for s in segs).strip()

    def _seg_suspicious(seg) -> bool:
        try:
            cr  = float(getattr(seg, "compression_ratio", 0.0) or 0.0)
            lp  = float(getattr(seg, "avg_logprob", -10.0) or -10.0)
            nsp = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
            return (cr > 2.1) or (lp < -0.65 and nsp > 0.45)
        except Exception:
            return False

    suspicious = False
    if full_text and looks_like_outro(full_text):
        suspicious = True
    elif not full_text:
        suspicious = True
    else:
        bad_flags = [_seg_suspicious(sg) for sg in segs] if segs else []
        if bad_flags and (sum(bad_flags) >= max(1, len(bad_flags)//2) or bad_flags[-1]):
            suspicious = True

    if suspicious:
        segments2, info2 = fw_transcribe(
            samples,
            language=_effective_language(),
            beam_size=1,
            temperature=0.0,
            without_timestamps=True,
            condition_on_previous_text=False,
            no_speech_threshold=commit2_nst,
            compression_ratio_threshold=max(1.95, cr_commit - 0.05),
        )
        segs2 = list(segments2)
        alt = " ".join((s.text or "").strip() for s in segs2).strip()

        def _seg_suspicious_retry(seg) -> bool:
            try:
                cr  = float(getattr(seg, "compression_ratio", 0.0) or 0.0)
                lp  = float(getattr(seg, "avg_logprob", -10.0) or -10.0)
                nsp = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
                return (cr > 2.0) or (lp < -0.55 and nsp > 0.50)
            except Exception:
                return False

        bad2 = [_seg_suspicious_retry(sg) for sg in segs2] if segs2 else []
        looks_outro = bool(alt and looks_like_outro(alt))
        metrics_bad = bool(bad2 and (sum(bad2) >= max(1, len(bad2)//2) or bad2[-1]))
        if (not alt) or looks_outro or metrics_bad:
            return ""
        full_text = alt

    low = (full_text or "").lower()
    if len(low) <= 24 and any(w in low for w in ("музык", "аплодисмент", "барабан", "спасибо")):
        return ""
    try:
        full_text = strip_trailing_dot(full_text)
    except Exception:
        pass
    return full_text
