# -*- mode: python ; coding: utf-8 -*-

a = Analysis( # type: ignore
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'webrtcvad'
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ( # type: ignore
    a.pure,
)

exe = EXE( # type: ignore
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='geneograph_voix',
    exclude_binaries=True,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='NONE',
)

coll = COLLECT( # type: ignore
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='geneograph_voix',
)
