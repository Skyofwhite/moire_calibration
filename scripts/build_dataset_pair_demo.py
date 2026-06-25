from __future__ import annotations

import json
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps

OUT = Path("output")
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "synthcal_figure3": "https://ar5iv.org/html/2307.01013/assets/image/4.png",
    "freqformer_pic2": "https://raw.githubusercontent.com/xyLiu339/Freqformer/8f1dd7f2b9d5e6cb6ba985302c893c861b823ded/assets/pic2.png",
    "freqformer_pic3": "https://raw.githubusercontent.com/xyLiu339/Freqformer/8f1dd7f2b9d5e6cb6ba985302c893c861b823ded/assets/pic3.png",
}


def download(name: str, url: str) -> Path:
    path = OUT / f"{name}.png"
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)
    with Image.open(path) as image:
        image.verify()
    return path


def fit_with_label(image: Image.Image, label: str, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    label_h = 28
    inner = (size[0] - 12, size[1] - label_h - 12)
    thumb = ImageOps.contain(image.convert("RGB"), inner, Image.Resampling.LANCZOS)
    x = (size[0] - thumb.width) // 2
    y = label_h + (inner[1] - thumb.height) // 2
    canvas.paste(thumb, (x, y))
    draw.rectangle((0, 0, size[0], label_h), fill=(245, 245, 245))
    draw.text((8, 8), label, fill="black", font=font)
    return canvas


def save_contact_sheet(items: list[tuple[str, Image.Image]], path: Path, cols: int = 3) -> None:
    cell = (520, 340)
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cell[0] * cols, cell[1] * rows), (225, 225, 225))
    for index, (label, image) in enumerate(items):
        x = (index % cols) * cell[0]
        y = (index // cols) * cell[1]
        sheet.paste(fit_with_label(image, label, cell), (x, y))
    sheet.save(path, quality=94)


def grid_crops(image: Image.Image, prefix: str, cols: int, rows: int) -> list[tuple[str, Image.Image]]:
    w, h = image.size
    result: list[tuple[str, Image.Image]] = []
    for row in range(rows):
        for col in range(cols):
            x0 = round(col * w / cols)
            x1 = round((col + 1) * w / cols)
            y0 = round(row * h / rows)
            y1 = round((row + 1) * h / rows)
            crop = image.crop((x0, y0, x1, y1))
            label = f"{prefix}_{rows}x{cols}_r{row + 1}c{col + 1}"
            crop.save(OUT / f"{label}.png")
            result.append((label, crop))
    return result


def left_panel_candidates(image: Image.Image, prefix: str) -> list[tuple[str, Image.Image]]:
    w, h = image.size
    candidates: list[tuple[str, Image.Image]] = []
    for rows in (1, 2):
        for cols in (8, 9, 10):
            for row in range(rows):
                x0, x1 = 0, round(w / cols)
                y0 = round(row * h / rows)
                y1 = round((row + 1) * h / rows)
                crop = image.crop((x0, y0, x1, y1))
                label = f"{prefix}_left_1of{cols}_row{row + 1}of{rows}"
                crop.save(OUT / f"{label}.png")
                candidates.append((label, crop))
    return candidates


def main() -> None:
    downloaded = {name: download(name, url) for name, url in SOURCES.items()}
    images = {name: Image.open(path).convert("RGB") for name, path in downloaded.items()}

    metadata = {
        name: {
            "source_url": SOURCES[name],
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
        }
        for name, image in images.items()
    }
    (OUT / "source_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    save_contact_sheet(list(images.items()), OUT / "00_source_overview.png", cols=3)

    synth_candidates: list[tuple[str, Image.Image]] = []
    for cols, rows in ((4, 2), (3, 2), (4, 3)):
        synth_candidates.extend(grid_crops(images["synthcal_figure3"], "synth", cols, rows))
    save_contact_sheet(synth_candidates, OUT / "01_synthcal_crop_candidates.png", cols=4)

    fhd_candidates: list[tuple[str, Image.Image]] = []
    fhd_candidates.extend(left_panel_candidates(images["freqformer_pic2"], "pic2"))
    fhd_candidates.extend(left_panel_candidates(images["freqformer_pic3"], "pic3"))
    save_contact_sheet(fhd_candidates, OUT / "02_fhdmi_input_candidates.png", cols=4)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
