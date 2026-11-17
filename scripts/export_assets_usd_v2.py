"""
Export each named asset to its own .usdc file, containing ONLY that asset.
Run in Blender 4.5: Scripting workspace -> New -> paste -> Run Script.
Watch progress in Window -> Toggle System Console.

What it does:
1. Sets semantic_class labels on the 5 dataset classes.
2. For each name in ASSETS: hides everything else, selects only that object
   (and its children), and exports it.
3. Opens each exported file and counts the meshes inside, so you can see
   immediately if a file still contains extra objects.
"""
import bpy
import os

# ---------------- settings ----------------
OUT_DIR = os.path.join(bpy.path.abspath("//"), "usd_assets")

# Blender object name -> output file name (without .usdc)
ASSETS = {
    # dataset classes
    "cardbox_A":         "cardbox_A",
    "SM_Crate_Big":      "SM_Crate_Big",
    "wooden_box":        "wooden_box",
    "plastic_container": "plastic_container",
    "pallet2v1_LOD0":    "pallet2v1_LOD0",
    # distractors
    "forklift":          "forklift",
    "Barrel_01":         "Barrel_01",
    "portable_generator": "portable_generator",
    # building
    "floor":             "floor",
    "roof":              "roof",
    "worn_metal_rack":   "worn_metal_rack",
    "wall_standard_standard_01":     "wall_standard_standard_01",
    "wall_standard_corner_large_01": "wall_standard_corner_large_01",
    "wall_pier_standard_01":         "wall_pier_standard_01",
    "wall_pier_corner_01":           "wall_pier_corner_01",
    "wall_door_garage_door_01":      "wall_door_garage_door_01",
    "door_garage_door_01":           "door_garage_door_01",
    "wall_door_garage_double_01":    "wall_door_garage_double_01",
    "door_garage_double_01":         "door_garage_double_01",
    "wall_door_recessed_small_01":   "wall_door_recessed_small_01",
    "door_recessed_small_01":        "door_recessed_small_01",
    "wall_window_tall_large_01":     "wall_window_tall_large_01",
    "window_tall_large_01":          "window_tall_large_01",
    "wall_window_centered_large_01": "wall_window_centered_large_01",
    "window_centered_large_01":      "window_centered_large_01",
}

# Dataset labels (object name -> class)
LABELS = {
    "cardbox_A":         "cardboard_box",
    "SM_Crate_Big":      "crate",
    "wooden_box":        "wooden_box",
    "plastic_container": "plastic_container",
    "pallet2v1_LOD0":    "pallet",
}
# ------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

if bpy.context.object and bpy.context.object.mode != "OBJECT":
    bpy.ops.object.mode_set(mode="OBJECT")

view_layer = bpy.context.view_layer


def find_object(name):
    """Exact name first, then case-insensitive."""
    obj = bpy.data.objects.get(name)
    if obj:
        return obj
    for o in bpy.data.objects:
        if o.name.lower() == name.lower():
            return o
    return None


def supported_options(wanted):
    """Keep only export options this Blender version understands."""
    props = bpy.ops.wm.usd_export.get_rna_type().properties.keys()
    return {k: v for k, v in wanted.items() if k in props}


def count_meshes(path):
    """Count Mesh prims in an exported file (uses Blender's bundled USD)."""
    try:
        from pxr import Usd
        stage = Usd.Stage.Open(path)
        return sum(1 for p in stage.Traverse() if p.GetTypeName() == "Mesh")
    except Exception as e:
        print(f"   (could not verify: {e})")
        return None


# 1. Labels
for name, label in LABELS.items():
    obj = find_object(name)
    if obj:
        obj["semantic_class"] = label
        print(f"Label: {obj.name} -> {label}")
    else:
        print(f"Label SKIPPED, object not found: {name}")

# Remember visibility so we can restore it afterwards
all_objs = list(view_layer.objects)
was_hidden = {o.name: o.hide_get() for o in all_objs}

missing, problems, done = [], [], 0

# 2. Export one asset at a time
for obj_name, file_name in ASSETS.items():
    obj = find_object(obj_name)
    if obj is None:
        missing.append(obj_name)
        continue

    keep = {obj} | set(obj.children_recursive)

    for o in all_objs:
        o.hide_set(o not in keep)
        o.select_set(o in keep)
    view_layer.objects.active = obj

    path = os.path.join(OUT_DIR, file_name + ".usdc")
    options = supported_options(dict(
        filepath=path,
        selected_objects_only=True,
        visible_objects_only=True,
        export_materials=True,
        export_custom_properties=True,   # keeps semantic_class
        export_textures_mode="NEW",      # copy textures next to the file
        export_lights=False,
        export_cameras=False,
        convert_world_material=False,    # no extra dome light per file
        export_animation=False,
    ))
    bpy.ops.wm.usd_export(**options)

    if obj.location.length > 0.001:
        print(f"WARNING: {obj.name} is not at 0,0,0")

    n = count_meshes(path)
    expected = sum(1 for o in keep if o.type == "MESH")
    if n is not None and n > expected:
        problems.append(f"{file_name}: {n} meshes (expected {expected})")
        print(f"PROBLEM {file_name}.usdc has {n} meshes, expected {expected}")
    else:
        print(f"OK      {file_name}.usdc ({n} mesh)")
    done += 1

# 3. Restore the scene
for o in all_objs:
    o.hide_set(was_hidden.get(o.name, False))
    o.select_set(False)

print("\n================ SUMMARY ================")
print(f"Exported {done} files to {OUT_DIR}")
if missing:
    print(f"Objects NOT FOUND (check names in Outliner): {missing}")
if problems:
    print("Files with extra objects:")
    for p in problems:
        print("  ", p)
else:
    print("All files contain only their own asset.")
