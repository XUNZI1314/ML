from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


REPO_ROOT = Path(__file__).resolve().parent
ASSET_DIR = REPO_ROOT / "assets"


def _build_gradient(size: int) -> Image.Image:
    image = Image.new("RGBA", (size, size))
    pixels = image.load()
    for y in range(size):
        for x in range(size):
            tx = x / max(1, size - 1)
            ty = y / max(1, size - 1)
            r = int(11 + 18 * tx + 10 * ty)
            g = int(81 + 90 * (1.0 - ty) + 20 * tx)
            b = int(114 + 70 * tx + 35 * (1.0 - ty))
            pixels[x, y] = (r, g, b, 255)
    return image


def _build_icon(size: int = 512) -> Image.Image:
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    gradient = _build_gradient(size)

    shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    pad = int(size * 0.07)
    shadow_draw.rounded_rectangle(
        [pad, pad + int(size * 0.02), size - pad, size - pad + int(size * 0.02)],
        radius=int(size * 0.22),
        fill=(5, 20, 28, 150),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=max(4, size // 48)))
    canvas.alpha_composite(shadow)

    mask = Image.new("L", (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle(
        [pad, pad, size - pad, size - pad],
        radius=int(size * 0.22),
        fill=255,
    )
    rounded = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    rounded.paste(gradient, (0, 0), mask)
    canvas.alpha_composite(rounded)

    draw = ImageDraw.Draw(canvas)
    inset = int(size * 0.15)
    outline_width = max(4, size // 48)

    draw.rounded_rectangle(
        [inset, inset, size - inset, size - inset],
        radius=int(size * 0.16),
        outline=(228, 247, 242, 220),
        width=outline_width,
    )

    left_circle = [
        int(size * 0.21),
        int(size * 0.23),
        int(size * 0.41),
        int(size * 0.43),
    ]
    right_circle = [
        int(size * 0.58),
        int(size * 0.23),
        int(size * 0.78),
        int(size * 0.43),
    ]
    draw.ellipse(left_circle, fill=(224, 248, 245, 210))
    draw.ellipse(right_circle, fill=(193, 238, 233, 170))
    draw.line(
        [(int(size * 0.34), int(size * 0.33)), (int(size * 0.65), int(size * 0.33))],
        fill=(242, 250, 248, 220),
        width=max(4, size // 54),
    )

    for offset in [0.0, 0.06, 0.12]:
        x0 = int(size * (0.28 + offset))
        y0 = int(size * 0.48)
        x1 = int(size * (0.62 + offset))
        y1 = int(size * 0.74)
        draw.arc(
            [x0, y0, x1, y1],
            start=200,
            end=340,
            fill=(232, 249, 246, 230),
            width=max(4, size // 56),
        )

    try:
        font = ImageFont.truetype("arialbd.ttf", size=int(size * 0.25))
    except OSError:
        font = ImageFont.load_default()

    text = "ML"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    text_x = (size - text_w) // 2
    text_y = int(size * 0.58) - text_h // 2

    draw.text(
        (text_x + max(2, size // 170), text_y + max(2, size // 170)),
        text,
        font=font,
        fill=(4, 30, 40, 110),
    )
    draw.text(
        (text_x, text_y),
        text,
        font=font,
        fill=(250, 255, 252, 245),
    )

    return canvas


def generate_brand_assets() -> dict[str, Path]:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    icon = _build_icon(size=512)
    png_path = ASSET_DIR / "app_icon.png"
    ico_path = ASSET_DIR / "app_icon.ico"
    icon.save(png_path, format="PNG")
    icon.save(ico_path, format="ICO", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    return {
        "png": png_path,
        "ico": ico_path,
    }


def main() -> int:
    outputs = generate_brand_assets()
    for key, path in outputs.items():
        print(f"{key}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
