# geneograph_voix.spec
# PyInstaller ≥ 6.x
import os
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules, collect_data_files

# Paths (do NOT use __file__ here)
SPEC_DIR   = os.path.abspath(os.getcwd())         # spec is run from its own folder
APP_DIR    = SPEC_DIR                              # main.py sits beside this spec
RTHOOKS_DIR= os.path.join(APP_DIR, "src/geneograph_voix/rthooks")      # <-- your custom runtime hooks
ASSETS_DIR = os.path.join(APP_DIR, "src/geneograph_voix/assets")
ICO_PATH    = os.path.join(ASSETS_DIR, "app.ico")
PNG_PATH    = os.path.join(ASSETS_DIR, "app_256.png") 

# Collect libs / data we need
hiddenimports  = []
hiddenimports += collect_submodules('ctranslate2')
hiddenimports += collect_submodules('faster_whisper')
hiddenimports += collect_submodules('tokenizers')

# Ensure Xet plugin is bundled
hiddenimports += ['hf_xet']

# Explicit hidden imports for dynamic dependencies
hiddenimports += ['requests']

binaries  = []
binaries += collect_dynamic_libs('ctranslate2')
binaries += collect_dynamic_libs('tokenizers')

datas  = []
datas += collect_data_files('tksheet', include_py_files=False)
# include plugin packages' data so importlib can find them at runtime
datas += collect_data_files('hf_xet', include_py_files=True)

# --- IMPORTANT: ship icon files for Tk window icon at runtime ---
if os.path.exists(ICO_PATH):
    datas.append((ICO_PATH, '.'))     # place at dist root as "app.ico"
if os.path.exists(PNG_PATH):
    datas.append((PNG_PATH, '.'))     # place at dist root as "app_256.png"

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[APP_DIR],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hooksconfig={},
    # webrtc removed, keep others excluded
    excludes=['onnx', 'onnxruntime', 'webrtcvad', 'pyaudio'],
    noarchive=False,
    hookspath=[RTHOOKS_DIR],     # <-- make sure PyInstaller sees your custom hook(s)
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='geneograph_voix',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # keep console for logs; set False to hide
    icon=ICO_PATH,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='geneograph_voix',
)
