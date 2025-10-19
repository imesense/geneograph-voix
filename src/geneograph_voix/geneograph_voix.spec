# geneograph_voix.spec
# PyInstaller ≥ 6.x
import os
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules, collect_data_files

# Paths (do NOT use __file__ here)
SPEC_DIR   = os.path.abspath(os.getcwd())              # spec is run from its own folder
APP_DIR    = SPEC_DIR                                  # main.py sits beside this spec
ASSETS_DIR = os.path.join(APP_DIR, "assets")
ICO_PATH   = os.path.join(ASSETS_DIR, "app.ico")

# Collect libs / data we need
hiddenimports  = []
hiddenimports += collect_submodules('ctranslate2')
hiddenimports += collect_submodules('faster_whisper')
hiddenimports += collect_submodules('tokenizers')

binaries  = []
binaries += collect_dynamic_libs('ctranslate2')
binaries += collect_dynamic_libs('tokenizers')

datas  = []
datas += collect_data_files('tksheet', include_py_files=False)

# Ship default JSONs if you have them locally
for fname in ('glossary.json', 'settings.json'):
    fpath = os.path.join(APP_DIR, fname)
    if os.path.exists(fpath):
        datas.append((fpath, '.'))

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[APP_DIR],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hooksconfig={},
    # Keep onnx/onnxruntime out (we removed VAD filter that needed it)
    excludes=['onnx', 'onnxruntime', 'webrtcvad', 'pyaudio'],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Set console=False to hide console; True is handy for logs while testing
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
    console=True,
    icon=ICO_PATH if os.path.exists(ICO_PATH) else None,
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
