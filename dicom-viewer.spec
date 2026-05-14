# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the DICOM Viewer.

Build:
    pyinstaller dicom-viewer.spec

Produces dist/dicom-viewer/ (Linux/Windows directory bundle) or
dist/DICOM Viewer.app (macOS .app bundle when sys.platform == 'darwin').
"""
import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None
APP_NAME = "dicom-viewer"
MAC_BUNDLE_NAME = "DICOM Viewer"
# Per-platform icon for the bundled executable / .app. SVG is the source;
# scripts/make_icons.py turns it into ICO / ICNS / PNG.
if sys.platform == "win32":
    APP_ICON = "assets/icon.ico"
elif sys.platform == "darwin":
    APP_ICON = "assets/icon.icns"
else:
    APP_ICON = "assets/icon.png"

# --- hidden imports ---------------------------------------------------------
# VTK loads many of its modules dynamically and PyInstaller misses some of
# them. Collect everything under vtkmodules.* to be safe.
hiddenimports: list[str] = []
hiddenimports += collect_submodules("vtkmodules")
hiddenimports += collect_submodules("dicom_viewer")

# pylibjpeg plugins are wired up via entry points; pull them in explicitly so
# pydicom can decode JPEG / JPEG-2000 / JPEG-LS DICOMs out of the box.
for plug in ("pylibjpeg_libjpeg", "pylibjpeg_openjpeg"):
    try:
        hiddenimports += collect_submodules(plug)
    except Exception:
        # Plugin not installed — pydicom will fall back to less codecs.
        pass

# torch + transformers (MedSAM) — both load many submodules lazily so the
# static analyser misses them. collect_submodules pulls the whole tree;
# excludes the parts that obviously aren't needed at runtime (tests etc.).
for pkg in ("torch", "transformers", "tokenizers", "safetensors", "huggingface_hub"):
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception:
        pass

# SimpleITK ships its native libs alongside the Python package; PyInstaller
# normally picks these up but collect_data_files makes sure data files (like
# transform XMLs) come along.
datas: list[tuple[str, str]] = []
# Ship the icon assets inside the bundle so the runtime can load them via
# sys._MEIPASS / assets/icon.png. Keeps the in-app window icon working even
# when launched from the installed location with no source checkout.
if os.path.isdir("assets"):
    datas.append(("assets/icon.png", "assets"))
    datas.append(("assets/icon.svg", "assets"))
for pkg in (
    "vtkmodules",
    "SimpleITK",
    "pydicom",
    "pylibjpeg",
    "torch",
    "transformers",
    "tokenizers",
):
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass

a = Analysis(
    ["src/dicom_viewer/__main__.py"],
    pathex=["src"],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # Tests should never end up inside the bundle.
        "pytest",
        "pytest_qt",
        "tests",
    ],
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
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,        # UPX has historically corrupted Qt plugins; keep it off.
    console=False,    # GUI app: no terminal window on Windows.
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=APP_ICON if os.path.exists(APP_ICON) else None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)

# On macOS, wrap the COLLECT directory into a proper .app bundle so it shows up
# as a single draggable item in /Applications and gets a Dock icon.
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{MAC_BUNDLE_NAME}.app",
        icon=APP_ICON if os.path.exists(APP_ICON) else None,
        bundle_identifier="com.dicomviewer.app",
        info_plist={
            "CFBundleName": MAC_BUNDLE_NAME,
            "CFBundleDisplayName": MAC_BUNDLE_NAME,
            "CFBundleShortVersionString": "0.1.0",
            "CFBundleVersion": "0.1.0",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
            "CFBundleDocumentTypes": [
                {
                    "CFBundleTypeName": "DICOM Viewer Project",
                    "CFBundleTypeExtensions": ["dcmproj"],
                    "CFBundleTypeRole": "Editor",
                    "LSItemContentTypes": ["com.dicomviewer.project"],
                }
            ],
            "UTExportedTypeDeclarations": [
                {
                    "UTTypeIdentifier": "com.dicomviewer.project",
                    "UTTypeDescription": "DICOM Viewer project",
                    "UTTypeConformsTo": ["public.data"],
                    "UTTypeTagSpecification": {
                        "public.filename-extension": ["dcmproj"],
                    },
                }
            ],
        },
    )
