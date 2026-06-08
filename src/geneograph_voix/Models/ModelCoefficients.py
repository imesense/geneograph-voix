# ---------------------------
# Dynamic VAD strictness (1..5)
# ---------------------------
VAD_STRICTNESS = 3
def _silero_params_for(level: int):
    level = int(max(1, min(5, level)))
    table = {
        1: dict(th=0.30, min_speech=80,  min_silence=150, pad=200, energy_db=-52.0),
        2: dict(th=0.35, min_speech=100, min_silence=200, pad=220, energy_db=-48.0),
        3: dict(th=0.40, min_speech=140, min_silence=260, pad=250, energy_db=-45.0),
        4: dict(th=0.48, min_speech=200, min_silence=320, pad=260, energy_db=-42.0),
        5: dict(th=0.56, min_speech=260, min_silence=380, pad=280, energy_db=-40.0),
    }
    return table[level]

def _energy_gate_db() -> float:
    return _silero_params_for(VAD_STRICTNESS)["energy_db"]

def _no_speech_thresholds():
    l = int(max(1, min(5, VAD_STRICTNESS)))
    preview_nst = {1:0.70, 2:0.65, 3:0.60, 4:0.58, 5:0.55}[l]
    commit1_nst = {1:0.65, 2:0.62, 3:0.58, 4:0.56, 5:0.54}[l]
    commit2_nst = {1:0.62, 2:0.60, 3:0.56, 4:0.54, 5:0.52}[l]
    cr_preview  = {1:2.40, 2:2.30, 3:2.20, 4:2.10, 5:2.05}[l]
    cr_commit   = {1:2.20, 2:2.10, 3:2.00, 4:1.95, 5:1.90}[l]
    return preview_nst, commit1_nst, commit2_nst, cr_preview, cr_commit

# ---------------------------
# Glossary strictness (1..5)
# ---------------------------
GLOSSARY_STRICTNESS = 3
