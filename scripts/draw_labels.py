"""
Draw YOLO boxes on dataset images so you can check the labels by eye.
Run with normal Python (pip install pillow):
    python draw_labels.py C:/path/to/warehouse_dataset_v1
Output goes to <dataset>/preview/.
"""
import os
import sys
from PIL import Image, ImageDraw

CLASSES = ["cardboard_box", "crate", "wooden_box", "plastic_container", "pallet"]
COLORS = ["#ffd400", "#ff5a1f", "#3fd17a", "#3aa0ff", "#e040fb"]
MAX_IMAGES = 30

root = sys.argv[1] if len(sys.argv) > 1 else "."
img_dir, lbl_dir = os.path.join(root, "images"), os.path.join(root, "labels")
out_dir = os.path.join(root, "preview")
os.makedirs(out_dir, exist_ok=True)

for name in sorted(os.listdir(img_dir))[:MAX_IMAGES]:
    stem = os.path.splitext(name)[0]
    img = Image.open(os.path.join(img_dir, name)).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    lbl = os.path.join(lbl_dir, stem + ".txt")
    n = 0
    if os.path.exists(lbl):
        for line in open(lbl).read().split("\n"):
            if not line.strip():
                continue
            c, cx, cy, bw, bh = line.split()
            c = int(c)
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            box = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
            draw.rectangle(box, outline=COLORS[c], width=2)
            draw.text((box[0] + 3, box[1] + 2), CLASSES[c], fill=COLORS[c])
            n += 1
    img.save(os.path.join(out_dir, name))
    print(f"{name}: {n} boxes")
print(f"Previews saved in {out_dir}")
