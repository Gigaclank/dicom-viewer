"""Generate raster icon assets from the logo design.

Produces:
  assets/icon-{16,32,48,64,128,256,512}.png  — individual sizes
  assets/icon.png                            — the 512 px master
  assets/icon.ico                            — multi-resolution Windows icon
  assets/icon.icns                           — macOS app icon

Run:
    .venv/bin/python scripts/make_icons.py

The output files are committed to the repo so build steps don't need to
re-run this. Re-run only when the logo design changes.

The PIL-drawn geometry matches assets/icon.svg pixel-for-pixel within
each raster's coordinate system.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent.parent
ASSETS = REPO / "assets"

BG = (13, 63, 77)          # dark teal — matches #0d3f4d in icon.svg
SLICE = (127, 220, 255)    # light blue — #7fdcff
CUBE_STROKE = (255, 255, 255)


def draw_logo(size: int) -> Image.Image:
    """Render the logo at `size` x `size` pixels."""
    s = size
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    def f(v: float) -> int:
        """Convert SVG fractional coordinate (0..512) to current size."""
        return int(round(v * s / 512.0))

    # Rounded background.
    d.rounded_rectangle((0, 0, s - 1, s - 1), radius=f(76), fill=BG)

    # Slice stack — five bars, slight rightward shift per row to suggest 3D.
    bars = [
        (82,  138, 238, 160),
        (86,  186, 242, 208),
        (90,  234, 246, 256),
        (94,  282, 250, 304),
        (98,  330, 254, 352),
    ]
    for x0, y0, x1, y1 in bars:
        d.rounded_rectangle(
            (f(x0), f(y0), f(x1), f(y1)),
            radius=max(1, f(6)),
            fill=SLICE,
        )

    # Arrow.
    arrow = [
        (268, 244), (312, 244), (312, 230),
        (352, 256), (312, 282), (312, 268), (268, 268),
    ]
    d.polygon([(f(x), f(y)) for x, y in arrow], fill=SLICE)

    # Cube — three faces (front, top, right).
    cube_stroke = max(2, f(9))
    cube_faces = [
        # front
        ([(360, 218), (446, 218), (446, 318), (360, 318)], (26, 71, 86)),
        # top
        ([(360, 218), (396, 178), (482, 178), (446, 218)], (52, 95, 110)),
        # right
        ([(446, 218), (482, 178), (482, 278), (446, 318)], (21, 65, 79)),
    ]
    for points, fill in cube_faces:
        d.polygon([(f(x), f(y)) for x, y in points], fill=fill, outline=None)
    # Outlines on top so the seams are sharp.
    for points, _fill in cube_faces:
        pts = [(f(x), f(y)) for x, y in points]
        d.line(pts + [pts[0]], fill=CUBE_STROKE, width=cube_stroke, joint="curve")

    return img


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)

    sizes = [16, 32, 48, 64, 128, 256, 512]
    rasters: dict[int, Image.Image] = {}
    for sz in sizes:
        img = draw_logo(sz)
        path = ASSETS / f"icon-{sz}.png"
        img.save(path)
        rasters[sz] = img
        print(f"wrote {path}")

    # The "main" 512 px PNG that Linux .desktop / general use refers to.
    (ASSETS / "icon.png").write_bytes((ASSETS / "icon-512.png").read_bytes())
    print(f"wrote {ASSETS / 'icon.png'}")

    # Multi-resolution Windows ICO. PIL packs the listed sizes into one file.
    ico_path = ASSETS / "icon.ico"
    rasters[256].save(
        ico_path,
        sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    print(f"wrote {ico_path}")

    # macOS ICNS. PIL's icns writer expects the high-res image; it bakes
    # multiple sub-images itself.
    icns_path = ASSETS / "icon.icns"
    try:
        rasters[512].save(icns_path)
        print(f"wrote {icns_path}")
    except Exception as e:
        print(f"WARN: could not write {icns_path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
