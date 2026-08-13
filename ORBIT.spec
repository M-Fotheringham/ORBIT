# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata

repository_root = Path(globals().get('SPECPATH', '.')).resolve()
cellpose_model = repository_root / 'build' / 'cellpose_models' / 'cpsam_v2'
if not cellpose_model.is_file():
    raise SystemExit(
        'The staged cpsam_v2 model is missing. Run:\n'
        '  uv run python scripts/stage_cellpose_model.py\n'
        'then build ORBIT again.'
    )

datas = [
    ('docs/figs/icon_logo.ico', 'docs/figs'),
    (str(cellpose_model), 'cellpose_models'),
    (
        'third_party/CELLPOSE_SAM_MODEL_NOTICE.txt',
        'licenses',
    ),
]
binaries = []
hiddenimports = []
datas += collect_data_files('orbit')
datas += copy_metadata('torch')
datas += copy_metadata('torchvision')
datas += copy_metadata('cellpose')
datas += copy_metadata('napari')
datas += copy_metadata('npe2')
binaries += collect_dynamic_libs('torch')
binaries += collect_dynamic_libs('torchvision')
tmp_ret = collect_all('cellpose')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('napari')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('npe2')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('vispy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('imagecodecs')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['src/orbit/app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ORBIT',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['docs/figs/icon_logo.ico'],
    contents_directory='.',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='ORBIT',
)
