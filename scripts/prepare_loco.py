"""
Prepare the real LOCO dataset for Experiments C and D, and turn your 42 test
photos into a labeled real-world test set.

What it does
  1. Reads LOCO's COCO-format annotation JSON(s)
  2. Maps LOCO classes to yours:
        pallet              -> pallet (4)
        small load carrier  -> plastic_container (3)
        stillage, forklift, pallet truck -> dropped (not your classes)
  3. TEST:  your 42 photos + their LOCO labels   -> OUT/loco_test
  4. TRAIN: all other annotated LOCO images       -> OUT/loco_train, OUT/loco_val
     LOCO images are video frames, so frames next to a test photo are almost
     the same picture. Those neighbors are left out of train/val (--neighbors).
  5. Writes YAML files for the experiments (see the end of this file's output)

Run with normal Python (pillow is only needed if the JSON lacks image sizes):
  python prepare_loco.py --loco-json E:/LOCO/loco-all-v1.json ^
                         --loco-images E:/LOCO ^
                         --test-dir E:/Sythetic_Data/Isaacsim/realphoto ^
                         --out E:/Sythetic_Data/Isaacsim/loco_prepared
"""
import argparse
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict

CLASSES = ["cardboard_box", "crate", "wooden_box", "plastic_container", "pallet"]
PALLET, CONTAINER = 4, 3
IMG_EXT = (".jpg", ".jpeg", ".png")

ap = argparse.ArgumentParser()
ap.add_argument("--loco-json", nargs="+", required=True,
                help="LOCO COCO annotation file(s)")
ap.add_argument("--loco-images", required=True,
                help="folder that contains the LOCO images (searched recursively)")
ap.add_argument("--test-dir", required=True, help="folder with your 42 test photos")
ap.add_argument("--out", required=True)
ap.add_argument("--synthetic", default="E:/Sythetic_Data/Isaacsim/OutIsaac_v2",
                help="synthetic dataset root (already split into train/ and val/)")
ap.add_argument("--neighbors", type=int, default=15,
                help="frames before/after each test photo to leave out of training")
ap.add_argument("--val", type=float, default=0.1, help="share of LOCO train for val")
ap.add_argument("--max-train", type=int, default=0, help="limit LOCO train images (0 = all)")
ap.add_argument("--no-slc", action="store_true",
                help="do NOT map small load carriers to plastic_container")
ap.add_argument("--min-box", type=float, default=4, help="drop boxes smaller than this (px)")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()


def norm(name):
    return re.sub(r"[_\-]+", " ", name.strip().lower())


def map_category(name):
    n = norm(name)
    if n == "pallet":
        return PALLET
    if not args.no_slc and ("load carrier" in n or n in ("slc", "klt")):
        return CONTAINER
    return None


def frame_key(fname):
    """Number in the file name, for ordering frames ('1564562567798,87' -> 1564562567798.87)."""
    stem = os.path.splitext(os.path.basename(fname))[0].replace(",", ".")
    m = re.search(r"\d+(\.\d+)?", stem)
    return float(m.group()) if m else 0.0


# ---------- 1. index image files on disk ----------
print("Indexing LOCO image files ...")
on_disk, dupes = {}, 0
for root, _, files in os.walk(args.loco_images):
    for f in files:
        if f.lower().endswith(IMG_EXT):
            if f in on_disk:
                dupes += 1
                continue
            on_disk[f] = os.path.join(root, f)
print(f"  {len(on_disk)} image files found" + (f" ({dupes} duplicate names ignored)" if dupes else ""))

# ---------- 2. read annotations ----------
records = {}           # basename -> dict(w, h, boxes=[(cls, x, y, w, h)])
cat_seen = Counter()
for jpath in args.loco_json:
    data = json.load(open(jpath, encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in data.get("categories", [])}
    imgs = {}
    for im in data.get("images", []):
        base = os.path.basename(im["file_name"])
        imgs[im["id"]] = base
        records.setdefault(base, dict(w=im.get("width"), h=im.get("height"), boxes=[]))
    for a in data.get("annotations", []):
        name = cats.get(a["category_id"], "?")
        cat_seen[name] += 1
        cls = map_category(name)
        if cls is None or a.get("iscrowd", 0):
            continue
        base = imgs.get(a["image_id"])
        if base is None:
            continue
        records[base]["boxes"].append((cls, *a["bbox"]))

print("\nLOCO categories (annotations):")
for name, n in cat_seen.most_common():
    cls = map_category(name)
    print(f"  {name:22s} {n:7d}  -> {CLASSES[cls] if cls is not None else 'dropped'}")
if not any(map_category(n) == PALLET for n in cat_seen):
    raise SystemExit("STOP: no 'pallet' category found. Check that --loco-json is the LOCO annotation file.")


# ---------- 3. helpers ----------
def image_size(path, rec):
    if rec["w"] and rec["h"]:
        return rec["w"], rec["h"]
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except ImportError:
        raise SystemExit("Image size missing in JSON; install pillow: pip install pillow")


def write_item(base, src, split_dir, counts):
    rec = records[base]
    w, h = image_size(src, rec)
    lines = []
    for cls, x, y, bw, bh in rec["boxes"]:
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(w, x + bw), min(h, y + bh)
        if x1 - x0 < args.min_box or y1 - y0 < args.min_box:
            continue
        lines.append(f"{cls} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} "
                     f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
        counts[cls] += 1
    dst = os.path.join(split_dir, "images", base)
    try:
        os.link(src, dst)                 # hard link: no extra disk space
    except OSError:
        shutil.copy2(src, dst)
    stem = os.path.splitext(base)[0]
    with open(os.path.join(split_dir, "labels", stem + ".txt"), "w") as f:
        f.write("\n".join(lines))


def fresh(split):
    d = os.path.join(args.out, split)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(os.path.join(d, "images"))
    os.makedirs(os.path.join(d, "labels"))
    return d


# ---------- 4. test set: your photos ----------
test_names = sorted(f for f in os.listdir(args.test_dir) if f.lower().endswith(IMG_EXT))
test_dir = fresh("loco_test")
test_counts = [0] * len(CLASSES)
not_annotated = []
for base in test_names:
    if base not in records:
        not_annotated.append(base)
        continue
    write_item(base, os.path.join(args.test_dir, base), test_dir, test_counts)
print(f"\nTEST: {len(test_names) - len(not_annotated)} of {len(test_names)} photos have LOCO labels")
if not_annotated:
    print("  no LOCO labels for (left out of the test set):")
    for b in not_annotated:
        print("   ", b)

# ---------- 5. train / val: everything else, minus near-duplicate frames ----------
by_folder = defaultdict(list)
for base, path in on_disk.items():
    by_folder[os.path.dirname(path)].append(base)
blocked = set(test_names)
for folder, names in by_folder.items():
    names.sort(key=frame_key)
    pos = {n: i for i, n in enumerate(names)}
    for t in test_names:
        if t in pos:
            i = pos[t]
            blocked.update(names[max(0, i - args.neighbors): i + args.neighbors + 1])

pool = sorted(b for b in records if b in on_disk and b not in blocked)
missing_files = sum(1 for b in records if b not in on_disk)
rng = random.Random(args.seed)
rng.shuffle(pool)
if args.max_train:
    pool = pool[:args.max_train]
n_val = int(len(pool) * args.val)
splits = {"loco_val": pool[:n_val], "loco_train": pool[n_val:]}
train_counts = {}
for split, names in splits.items():
    d = fresh(split)
    counts = [0] * len(CLASSES)
    for base in names:
        write_item(base, on_disk[base], d, counts)
    train_counts[split] = counts
print(f"\nTRAIN: {len(splits['loco_train'])}  VAL: {len(splits['loco_val'])}  "
      f"(left out as test neighbors: {len(blocked) - len(test_names)}, "
      f"annotated but file not found: {missing_files})")

# ---------- 6. YAML files ----------
out = args.out.replace("\\", "/")
syn = args.synthetic.replace("\\", "/")
names_line = f"nc: {len(CLASSES)}\nnames: {CLASSES}\n"
yamls = {
    # evaluate any model (v1, v2, fine-tuned) on your real photos
    "data_test.yaml": f"train: {out}/loco_train/images\nval: {out}/loco_test/images\n"
                      f"test: {out}/loco_test/images\n",
    # Experiment C: real data only
    "data_loco_only.yaml": f"train: {out}/loco_train/images\nval: {out}/loco_val/images\n"
                           f"test: {out}/loco_test/images\n",
    # Experiment D: synthetic + real together (keeps all 5 classes alive)
    "data_mix.yaml": f"train:\n  - {syn}/train/images\n  - {out}/loco_train/images\n"
                     f"val: {out}/loco_val/images\ntest: {out}/loco_test/images\n",
}
for fname, body in yamls.items():
    with open(os.path.join(args.out, fname), "w") as f:
        f.write(body + names_line)

print("\nBoxes per class:")
print(f"  {'':12s} " + " ".join(f"{c[:10]:>10s}" for c in CLASSES))
for split, c in [("loco_train", train_counts["loco_train"]),
                 ("loco_val", train_counts["loco_val"]), ("loco_test", test_counts)]:
    print(f"  {split:12s} " + " ".join(f"{v:10d}" for v in c))
print(f"\nYAML files written to {out}: " + ", ".join(yamls))
