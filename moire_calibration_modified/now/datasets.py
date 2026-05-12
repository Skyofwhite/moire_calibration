import os
from pathlib import Path
from PIL import Image, ImageDraw

input_dir = Path(r"C:\Users\USER\Desktop\project\calibration\moire\datasets\test")
output_dir = input_dir / "augmented_9x"
output_dir.mkdir(parents=True, exist_ok=True)

offsets = [(150, 0), (-150, 0), (0, 150)]
angles = [15, -15, 30]
box_ratio = 0.4
valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

images = [p for p in input_dir.iterdir() if p.suffix.lower() in valid_ext]
total = len(images)

print(f"[INFO] 총 이미지 수: {total}")

count_out = 0

for idx, img_path in enumerate(images, start=1):
    print(f"[{idx}/{total}] 처리 중: {img_path.name}")

    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    stem = img_path.stem

    for i, (dx, dy) in enumerate(offsets, 1):
        canvas = Image.new("RGB", (w, h), (0, 0, 0))
        canvas.paste(img, (dx, dy))
        canvas.save(output_dir / f"{stem}_translation_{i}.jpg", quality=95)
        count_out += 1

    for i, angle in enumerate(angles, 1):
        rotated = img.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor=(0, 0, 0)
        )
        rotated.save(output_dir / f"{stem}_rotation_{i}.jpg", quality=95)
        count_out += 1

    bw, bh = int(w * box_ratio), int(h * box_ratio)
    positions = [(0, 0), ((w - bw)//2, (h - bh)//2), (w - bw, h - bh)]

    for i, (x, y) in enumerate(positions, 1):
        masked = img.copy()
        draw = ImageDraw.Draw(masked)
        draw.rectangle([x, y, x + bw, y + bh], fill=(0, 0, 0))
        masked.save(output_dir / f"{stem}_blackbox_{i}.jpg", quality=95)
        count_out += 1

print(f"[DONE] 총 생성 이미지 수: {count_out}")
