# Releasing

CI runs on every push / PR (`.github/workflows/tests.yml`).

Releases are cut by pushing an annotated tag matching `v*`:

```bash
git tag -a v0.1.0 -m "Release 0.1.0"
git push origin v0.1.0
```

This triggers `.github/workflows/release.yml`, which:

1. **Linux** — bundles with PyInstaller, then uses `fpm` to wrap the output as
   `dicom-viewer_<version>_amd64.deb`. Installs to `/opt/dicom-viewer/` with a
   wrapper at `/usr/bin/dicom-viewer`, registers the `.dcmproj` MIME type, and
   places a `.desktop` entry in `/usr/share/applications/`.
2. **Windows** — bundles with PyInstaller, then runs the NSIS script at
   `scripts/dicom-viewer.nsi` to produce
   `dist/dicom-viewer-<version>-setup.exe`. Adds Start Menu and desktop
   shortcuts, registers `.dcmproj` in the registry, and creates an
   Add/Remove Programs entry.
3. **macOS** — bundles via PyInstaller's `BUNDLE` block (configured in
   `dicom-viewer.spec`) to produce `DICOM Viewer.app`, then `dmgbuild` wraps
   it into `dicom-viewer-<version>.dmg` with the standard
   drag-onto-Applications layout.
4. **publish-release** — downloads all three artifacts and creates a GitHub
   Release attached to the tag, with auto-generated notes.

## No code signing yet

The current pipeline ships unsigned binaries. Users will see:

- **Windows** — "Windows protected your PC" SmartScreen warning on first run.
  Click "More info" → "Run anyway".
- **macOS** — Gatekeeper will block the app on first launch. Right-click the
  app in Finder → Open, then confirm. Or run
  `xattr -dr com.apple.quarantine "/Applications/DICOM Viewer.app"`.
- **Linux** — no warnings.

Adding code signing later means:
- Windows: a code-signing cert + `signtool` step.
- macOS: an Apple Developer ID Application cert + `codesign` + `notarytool`
  submission + `xcrun stapler staple` on the DMG.

Both rely on certs stored as GitHub Actions secrets.

## Local test build

```bash
.venv/bin/pip install pyinstaller
.venv/bin/pyinstaller --noconfirm dicom-viewer.spec
./dist/dicom-viewer/dicom-viewer   # or open "dist/DICOM Viewer.app" on macOS
```

If PyInstaller misses a module at runtime, add it to `hiddenimports` in
`dicom-viewer.spec`.
