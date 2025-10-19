# prep_assets.py
import os, torch

ASSETS = os.path.join(os.path.abspath("."), "assets")
TARGET = os.path.join(ASSETS, "torch_cache")
os.makedirs(TARGET, exist_ok=True)
os.environ["TORCH_HOME"] = TARGET

print("Priming Silero VAD into", TARGET)
torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    trust_repo=True,
    force_reload=False,
)
print("Silero VAD cached.")
