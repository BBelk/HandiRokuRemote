# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['rokuRemote.py'],
    pathex=[],
    binaries=[],
    datas=[
        # Include only necessary mediapipe runtime files
        # For example, if you only need the 'hands' module:
        (r'C:\ProgramData\anaconda3\envs\rokuRemoteEnv\Lib\site-packages\mediapipe\modules', 'mediapipe/modules'),
        # Include additional Python files
        ('gestureDetection.py', '.'),
        ('tooltip.py', '.'),  # Include if used
        # Include images
        ('images/icon.ico', 'images'),
        ('images/tooltip-images', 'images/tooltip-images')
    ],

    hiddenimports=[
        'mediapipe.python._framework_bindings',  # For mediapipe bindings
        'mediapipe.python.solutions',            # To include hands solution
        'PIL._imagingtk',                        # For tkinter/Pillow
        'PIL.Image',                             # Pillow
        'cv2',                                   # OpenCV
        'numpy',                                 # NumPy
        'secrets'
        # Include any other hidden imports required
    ],

    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
    ],
    noarchive=False,
    optimize=0,  # Optimize bytecode
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='HandiRokuRemote',
    debug=False,  # Disable debug logging in final builds
    bootloader_ignore_signals=False,
    strip=True,   # Strip symbols to reduce size
    upx=True,     # Enable UPX compression
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Hide console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='images/icon.ico'
)

# Explicitly set UPX directory if needed
import os
os.environ['UPX_DIR'] = r'C:\Tools\upx'
