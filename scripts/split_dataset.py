"""
Split the synthetic dataset into train / val by SCENE (not by image),
so views of the same layout never end up in both sets.
Run with normal Python:
    python split_dataset.py E:/Sythetic_Data/Isaacsim/OutIsaac_v2
Optional real test set (a folder with images/ + labels/ in YOLO format):
    python split_dataset.py E:/.../OutIsaac_v2 --real-test E:/.../real_test

Safe to re-run: old train/ and val/ folders are rebuilt from scratch.
"""
import argparse
import json
import os
import random
import shutil

CLASSES = ["cardboard_box", "crate", "wooden_box", "plastic_container", "pallet"]

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--val", type=float, default=0.1, help="share of scenes for val")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--real-test", default=None)
args = ap.parse_args()

src_img, src_lbl = os.path.join(args.root, "images"), os.path.join(args.root, "labels")
images = sorted(f for f in os.listdir(src_img) if f.endswith(".png"))
labels = {os.path.splitext(f)[0] for f in os.listdir(src_lbl) if f.endswith(".txt")}

# 1. Every image must have a label file. A missing label would teach YOLO
#    that the objects in the image are background.
no_label = [f for f in images if os.path.splitext(f)[0] not in labels]
if no_label:
    print("STOP: these images have no label file (delete them or restore labels):")
    for f in no_label:
        print("  ", f)
    raise SystemExit(1)

# 2. Image count should match what the generator wrote (catches leftovers
#    from earlier test runs).
meta_path = os.path.join(args.root, "metadata.json")
if os.path.exists(meta_path):
    expected = json.load(open(meta_path)).get("images")
    if expected is not None and expected != len(images):
        print(f"WARNING: metadata.json says {expected} images, folder has {len(images)}. "
              "Check for leftover files from test runs.")

scenes = sorted({f.split("_")[0] for f in images})          # e.g. "s0042"
random.Random(args.seed).shuffle(scenes)
n_val = max(1, int(len(scenes) * args.val))
val_scenes = set(scenes[:n_val])

# 3. Rebuild train/ and val/ so files from an earlier split cannot linger
for split in ("train", "val"):
    shutil.rmtree(os.path.join(args.root, split), ignore_errors=True)
    for kind in ("images", "labels"):
        os.makedirs(os.path.join(args.root, split, kind))

counts = {"train": 0, "val": 0}
boxes = {"train": [0] * len(CLASSES), "val": [0] * len(CLASSES)}
for f in images:
    split = "val" if f.split("_")[0] in val_scenes else "train"
    stem = os.path.splitext(f)[0]
    shutil.copy2(os.path.join(src_img, f), os.path.join(args.root, split, "images", f))
    lbl = os.path.join(src_lbl, stem + ".txt")
    shutil.copy2(lbl, os.path.join(args.root, split, "labels", stem + ".txt"))
    for line in open(lbl).read().split("\n"):
        if line.strip():
            boxes[split][int(line.split()[0])] += 1
    counts[split] += 1

root = args.root.replace("\\", "/")
with open(os.path.join(args.root, "data.yaml"), "w") as f:
    f.write(f"path: {root}\ntrain: train/images\nval: val/images\n")
    if args.real_test:
        f.write(f"test: {args.real_test.replace(chr(92), '/')}/images\n")
    f.write(f"nc: {len(CLASSES)}\nnames: {CLASSES}\n")

print(f"train: {counts['train']} images, val: {counts['val']} images "
      f"({n_val} of {len(scenes)} scenes held out)")
for split in ("train", "val"):
    print(f"{split:5s} boxes:", dict(zip(CLASSES, boxes[split])))
print("data.yaml updated")
