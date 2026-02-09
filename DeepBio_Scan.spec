# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['src/entry_point.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('src/interface', 'src/interface'),
        ('src/edge', 'src/edge')
    ],
    hiddenimports=[
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'altair',
        'pyarrow',
        'lancedb',
        'polars',
        'sklearn.utils._typedefs',
        'sklearn.neighbors._partition_nodes'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DeepBio_Scan',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True, # We keep console=True so we can verify the "Boot Sequence" but in prod use False
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DeepBio_Scan',
)
