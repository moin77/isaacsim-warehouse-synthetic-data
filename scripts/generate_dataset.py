"""
Warehouse synthetic dataset generator v2 (Isaac Sim 5.x, Script Editor).

Run AFTER build_warehouse.py, in the same stage:
    Window -> Script Editor -> paste -> Run

v2 changes (to close the synthetic-to-real gap seen on real photos):
  - every labeled object can get a different look: tinted copies of its own
    material (keeps the texture), flat colors, or random CC0 textures
  - brighter, more varied lighting + random fill lights near objects
  - 3 camera modes: LOW (0.4-1.2 m, wide lens), FORKLIFT, DRONE
  - stacks of empty pallets, pallets seen from the side
  - more distractors: cabinets, drums/rolls, wall signs, random colors

Same SEED + same settings = same dataset.
"""
import asyncio
import datetime
import json
import math
import os
import random

import numpy as np
import omni.replicator.core as rep
import omni.usd
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

# ======================= settings: edit these =======================
ASSET_DIR = "E:/Sythetic_Data/tool-detector/Unreal Engine/USDfile"  # your .usdc folder
OUT_DIR = "E:/Sythetic_Data/Isaacsim/OutIsaac_v2"                    # new folder
TEXTURE_DIR = ""          # optional: folder of CC0 .jpg/.png textures (ambientCG,
                          # Poly Haven). Leave "" to use tints + flat colors only.
SEED = 42
NUM_SCENES = 200          # new prop layout per scene
VIEWS_PER_SCENE = 5       # camera views per layout -> 1000 images
RESOLUTION = (1280, 720)
RT_SUBFRAMES = 8          # render passes per image (higher = cleaner, slower)

# Warehouse size: must match build_warehouse.py
W, D, H = 30.0, 18.0, 9.0
WALL_MARGIN = 1.0         # keep props this far from walls

# Rack shelf levels (top surface Z), measured in Blender
SHELF_Z = [0.0, 0.500, 1.069, 1.639, 2.208]
SHELF_THICKNESS = 0.035
SHELF_X_MARGIN = 0.05
SHELF_Y_MARGIN = 0.03

MAX_OCCLUSION = 0.7       # drop boxes hidden more than 70%
MIN_BOX_PX = 12           # drop boxes smaller than this (pixels)

# Appearance randomization
P_ORIGINAL_LOOK = 0.25    # chance an object keeps its original material
TINTS_PER_CLASS = 10      # tinted copies of each class material
FLATS_PER_CLASS = 3       # plain-color variants per class

# Walls: real warehouses are mostly grey/white, not red brick
P_KEEP_BRICK = 0.35       # chance a scene keeps the brick walls
WALL_COLORS = [(0.85, 0.85, 0.83), (0.70, 0.71, 0.72), (0.55, 0.56, 0.58),
               (0.90, 0.88, 0.80), (0.62, 0.66, 0.70), (0.78, 0.76, 0.70)]

# Skip broken frames (camera inside an object, render not ready)
MIN_MEAN_BRIGHTNESS = 12  # 0-255
MIN_STD = 4

# Lighting
CEILING_INTENSITY = (150000, 350000)
DOME_INTENSITY = (800, 4000)
FILL_LIGHTS = (2, 6)      # random extra lights near objects per scene

# Camera modes: share of images
CAM_MODES = {"low": 0.40, "forklift": 0.35, "drone": 0.25}

# Dataset classes (YOLO ids follow this order)
CLASSES = ["cardboard_box", "crate", "wooden_box", "plastic_container", "pallet"]

# asset file name -> class (None = unlabeled distractor)
ASSETS = {
    "cardbox_A":          "cardboard_box",
    "SM_Crate_Big":       "crate",
    "wooden_box":         "wooden_box",
    "plastic_container":  "plastic_container",
    "pallet2v1_LOD0":     "pallet",
    "forklift":           None,
    "Barrel_01":          None,
    "portable_generator": None,
    "worn_metal_rack":    None,   # only measured, already placed by the shell
}

# Target colors for tinted variants (RGB 0-1) and how strongly to mix them in.
# Real-world look of each class: brown/white cardboard, light/grey/blue pallets,
# blue/red/grey/yellow plastic bins, light to dark wood.
PALETTES = {
    "cardbox_A": dict(mix=(0.4, 0.75), colors=[       # warm kraft brown, not olive
        (0.72, 0.52, 0.32), (0.62, 0.44, 0.26), (0.92, 0.90, 0.86),
        (0.80, 0.64, 0.44), (0.55, 0.42, 0.30)]),
    "pallet2v1_LOD0": dict(mix=(0.3, 0.75), colors=[
        (0.88, 0.76, 0.55), (0.80, 0.68, 0.50), (0.62, 0.60, 0.55),
        (0.50, 0.38, 0.24), (0.10, 0.30, 0.70)]),
    "plastic_container": dict(mix=(0.75, 0.95), colors=[   # high mix: no purple lids
        (0.05, 0.30, 0.75), (0.80, 0.10, 0.06), (0.55, 0.57, 0.60),
        (0.90, 0.72, 0.05), (0.10, 0.50, 0.25), (0.08, 0.08, 0.09)]),
    "SM_Crate_Big": dict(mix=(0.2, 0.6), colors=[
        (0.88, 0.76, 0.55), (0.70, 0.55, 0.35), (0.60, 0.58, 0.52),
        (0.45, 0.32, 0.20)]),
    "wooden_box": dict(mix=(0.2, 0.6), colors=[
        (0.40, 0.42, 0.25), (0.75, 0.62, 0.42), (0.55, 0.42, 0.28),
        (0.30, 0.30, 0.28)]),
}
# ===================================================================

stage = omni.usd.get_context().get_stage()
SHELL = "/World/Warehouse"
PROPS = "/World/Props"
LOOKS_ROOT = "/World/Looks"
CAM_PATH = "/World/DataCam"
F = Sdf.ValueTypeNames

if not stage.GetPrimAtPath(SHELL):
    raise RuntimeError("Run build_warehouse.py first: /World/Warehouse not found")


def asset_path(name):
    return os.path.join(ASSET_DIR, name + ".usdc").replace("\\", "/")


missing = [n for n in ASSETS if not os.path.exists(asset_path(n))]
if missing:
    raise FileNotFoundError(f"Missing in {ASSET_DIR}: {missing}")


# ---------------- semantic labels ----------------
def add_label(prim, cls):
    try:
        from isaacsim.core.utils.semantics import add_labels
        add_labels(prim, labels=[cls], instance_name="class")
    except ImportError:
        from isaacsim.core.utils.semantics import add_update_semantics
        add_update_semantics(prim, semantic_label=cls, type_label="class")


# ---------------- measure every asset, find its material and UV name ----------
def bbox_cache():
    return UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                             [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])


def find_surface(mat_prim):
    for p in Usd.PrimRange(mat_prim):
        if p.IsA(UsdShade.Shader):
            sh = UsdShade.Shader(p)
            if sh.GetIdAttr().Get() == "UsdPreviewSurface":
                return sh
    return None


INFO = {}
for name in ASSETS:
    probe = f"/World/_Probe/{name}"
    UsdGeom.Xform.Define(stage, probe)
    root = probe + "/asset"
    stage.DefinePrim(root).GetReferences().AddReference(asset_path(name))
    r = bbox_cache().ComputeWorldBound(stage.GetPrimAtPath(probe)).ComputeAlignedRange()
    mn, mx = r.GetMin(), r.GetMax()
    info = dict(cx=(mn[0] + mx[0]) / 2, cy=(mn[1] + mx[1]) / 2, zmin=mn[2],
                size=(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]),
                material=None, uv="st")
    layer = Sdf.Layer.FindOrOpen(asset_path(name))
    default = layer.defaultPrim if layer else ""
    for p in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if info["material"] is None and p.IsA(UsdShade.Material) and default:
            info["material"] = "/" + default + str(p.GetPath())[len(root):]
        if p.IsA(UsdGeom.Mesh):
            for pv in UsdGeom.PrimvarsAPI(p).GetPrimvars():
                if pv.GetTypeName() in (F.TexCoord2fArray, F.Float2Array):
                    info["uv"] = pv.GetPrimvarName()
                    break
    INFO[name] = info
stage.RemovePrim("/World/_Probe")
for n, i in INFO.items():
    print(f"{n:20s} size {[round(v, 2) for v in i['size']]}  uv={i['uv']}")


# ---------------- material variants ----------------
def make_flat(path, color, rough, metal=0.0):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", F.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("roughness", F.Float).Set(rough)
    sh.CreateInput("metallic", F.Float).Set(metal)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def make_textured(path, tex_file, uv_name, rough):
    mat = UsdShade.Material.Define(stage, path)
    st = UsdShade.Shader.Define(stage, path + "/uv")
    st.CreateIdAttr("UsdPrimvarReader_float2")
    st.CreateInput("varname", F.String).Set(uv_name)
    st.CreateOutput("result", F.Float2)
    tex = UsdShade.Shader.Define(stage, path + "/tex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", F.Asset).Set(tex_file)
    tex.CreateInput("st", F.Float2).ConnectToSource(st.ConnectableAPI(), "result")
    tex.CreateInput("wrapS", F.Token).Set("repeat")
    tex.CreateInput("wrapT", F.Token).Set("repeat")
    tex.CreateOutput("rgb", F.Float3)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", F.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    sh.CreateInput("roughness", F.Float).Set(rough)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def make_tint(path, name, color, mix, bright, rough):
    """Copy the asset's own material and recolor its texture:
    color_out = texture * scale + bias  (keeps print, wood grain, labels)."""
    info = INFO[name]
    if not info["material"]:
        return None
    prim = stage.DefinePrim(path)
    prim.GetReferences().AddReference(asset_path(name), info["material"])
    surf = find_surface(prim)
    if surf is None:
        stage.RemovePrim(path)
        return None
    scale = [(1.0 - mix) * bright] * 3
    bias = [c * mix for c in color]
    diffuse = surf.GetInput("diffuseColor")
    if diffuse and diffuse.HasConnectedSource():
        src = diffuse.GetConnectedSource()[0].GetPrim()
        tex = UsdShade.Shader(src)
        tex.CreateInput("scale", F.Float4).Set(Gf.Vec4f(*scale, 1.0))
        tex.CreateInput("bias", F.Float4).Set(Gf.Vec4f(*bias, 0.0))
    else:
        surf.CreateInput("diffuseColor", F.Color3f).Set(Gf.Vec3f(*color))
    rin = surf.GetInput("roughness")
    if not (rin and rin.HasConnectedSource()):
        surf.CreateInput("roughness", F.Float).Set(rough)
    return UsdShade.Material(prim)


def texture_files():
    if not TEXTURE_DIR or not os.path.isdir(TEXTURE_DIR):
        return []
    return sorted(os.path.join(TEXTURE_DIR, f).replace("\\", "/")
                  for f in os.listdir(TEXTURE_DIR)
                  if f.lower().endswith((".jpg", ".jpeg", ".png")))


if stage.GetPrimAtPath(LOOKS_ROOT):
    stage.RemovePrim(LOOKS_ROOT)
UsdGeom.Scope.Define(stage, LOOKS_ROOT)
mrng = random.Random(SEED + 7)
TEXTURES = texture_files()
LOOKS = {}
for name, pal in PALETTES.items():
    looks = []
    for k in range(TINTS_PER_CLASS):
        m = make_tint(f"{LOOKS_ROOT}/{name}_tint{k}", name, mrng.choice(pal["colors"]),
                      mrng.uniform(*pal["mix"]), mrng.uniform(0.9, 1.5),
                      mrng.uniform(0.45, 0.9))
        if m:
            looks.append(m)
    for k in range(FLATS_PER_CLASS):
        base = mrng.choice(pal["colors"])
        color = [min(1.0, max(0.0, c + mrng.uniform(-0.08, 0.08))) for c in base]
        looks.append(make_flat(f"{LOOKS_ROOT}/{name}_flat{k}", color, mrng.uniform(0.4, 0.9)))
    for k, tf in enumerate(mrng.sample(TEXTURES, min(4, len(TEXTURES)))):
        looks.append(make_textured(f"{LOOKS_ROOT}/{name}_tex{k}", tf, INFO[name]["uv"],
                                   mrng.uniform(0.5, 0.9)))
    LOOKS[name] = looks
    print(f"looks for {name}: {len(looks)}"
          + ("" if INFO[name]["material"] else "  (no material found: flat colors only)"))

DISTRACT_LOOKS = [make_flat(f"{LOOKS_ROOT}/distract{k}",
                            (mrng.random(), mrng.random(), mrng.random()),
                            mrng.uniform(0.2, 0.9), mrng.choice([0.0, 0.0, 0.6]))
                  for k in range(24)]


WALL_LOOKS = []
for k, c in enumerate(WALL_COLORS):
    WALL_LOOKS.append(make_flat(f"{LOOKS_ROOT}/wall{k}", c, mrng.uniform(0.6, 0.95)))
for k, tf in enumerate(mrng.sample(TEXTURES, min(4, len(TEXTURES)))):
    WALL_LOOKS.append(make_textured(f"{LOOKS_ROOT}/wall_tex{k}", tf, "st", 0.85))


def bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)


# Plain wall pieces (not doors/windows); make them non-instanced so each
# scene can recolor them.
WALL_PRIMS = []
walls_root = stage.GetPrimAtPath(f"{SHELL}/Walls")
if walls_root:
    for p in walls_root.GetChildren():
        if p.GetName().endswith(("_door", "_glass")):
            continue
        for c in p.GetChildren():
            c.SetInstanceable(False)
        WALL_PRIMS.append(p)
print(f"wall pieces that can be recolored: {len(WALL_PRIMS)}")


def randomize_walls(rng):
    if not WALL_PRIMS:
        return
    look = None if rng.random() < P_KEEP_BRICK else rng.choice(WALL_LOOKS)
    for p in WALL_PRIMS:
        api = UsdShade.MaterialBindingAPI.Apply(p)
        if look is None:
            api.UnbindAllBindings()
        else:
            api.Bind(look, bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    return "brick" if look is None else look.GetPath().name


# ---------------- placement helpers ----------------
class Occupancy:
    """2D floor rectangles already taken (x0, x1, y0, y1)."""
    def __init__(self):
        self.rects = []

    def free(self, x0, x1, y0, y1):
        return all(x1 <= a or x0 >= b or y1 <= c or y0 >= d
                   for a, b, c, d in self.rects)

    def add(self, x0, x1, y0, y1):
        self.rects.append((x0, x1, y0, y1))

    def inside(self, x, y, margin=0.0):
        return any(a - margin <= x <= b + margin and c - margin <= y <= d + margin
                   for a, b, c, d in self.rects)


def half_extents(w, d, yaw_deg, pad=0.05):
    t = math.radians(yaw_deg)
    c, s = abs(math.cos(t)), abs(math.sin(t))
    return (w * c + d * s) / 2 + pad, (w * s + d * c) / 2 + pad


class Scene:
    def __init__(self, rng):
        self.rng = rng
        self.n = 0
        self.occ = Occupancy()
        self.targets = []   # centers of labeled objects, for camera aiming

    def pick_look(self, name):
        looks = LOOKS.get(name)
        if not looks or self.rng.random() < P_ORIGINAL_LOOK:
            return None
        return self.rng.choice(looks)

    def place(self, name, x, y, z_bottom, yaw_deg, look="auto"):
        """Put the asset's bbox center at (x, y) and its bottom at z_bottom."""
        i = INFO[name]
        t = math.radians(yaw_deg)
        ox = i["cx"] * math.cos(t) - i["cy"] * math.sin(t)
        oy = i["cx"] * math.sin(t) + i["cy"] * math.cos(t)
        path = f"{PROPS}/{name}_{self.n:04d}"
        self.n += 1
        parent = UsdGeom.Xform.Define(stage, path)
        api = UsdGeom.XformCommonAPI(parent)
        api.SetTranslate(Gf.Vec3d(x - ox, y - oy, z_bottom - i["zmin"]))
        api.SetRotate(Gf.Vec3f(0.0, 0.0, yaw_deg))
        child = stage.DefinePrim(path + "/asset")
        child.GetReferences().AddReference(asset_path(name))
        cls = ASSETS[name]
        if cls:
            add_label(parent.GetPrim(), cls)
            self.targets.append((x, y, z_bottom + i["size"][2] / 2, cls))
            if look == "auto":
                look = self.pick_look(name)
            if look is not None:
                bind(parent.GetPrim(), look)
        else:
            child.SetInstanceable(True)   # distractors share GPU memory

    def find_rect(self, w, d, tries=40, pad=0.05):
        for _ in range(tries):
            yaw = self.rng.choice([0, 90, 180, 270]) + self.rng.uniform(-10, 10)
            hx, hy = half_extents(w, d, yaw, pad)
            x0, x1 = WALL_MARGIN + hx, W - WALL_MARGIN - hx
            y0, y1 = WALL_MARGIN + hy, D - WALL_MARGIN - hy
            if x0 >= x1 or y0 >= y1:
                return None
            x, y = self.rng.uniform(x0, x1), self.rng.uniform(y0, y1)
            if self.occ.free(x - hx, x + hx, y - hy, y + hy):
                self.occ.add(x - hx, x + hx, y - hy, y + hy)
                return x, y, yaw
        return None

    def find_floor_spot(self, name, tries=40, pad=0.05):
        w, d, _ = INFO[name]["size"]
        return self.find_rect(w, d, tries, pad)


# ---------------- what goes where ----------------
def read_racks():
    """Rack centers from the shell (racks are not rotated)."""
    racks = []
    root = stage.GetPrimAtPath(f"{SHELL}/Racks")
    if not root:
        return racks
    i = INFO["worn_metal_rack"]
    for p in root.GetChildren():
        m = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = m.ExtractTranslation()
        racks.append((t[0] + i["cx"], t[1] + i["cy"]))
    return racks


def add_racks_to_occupancy(sc, racks):
    rw, rd, _ = INFO["worn_metal_rack"]["size"]
    for x, y in racks:
        sc.occ.add(x - rw / 2 - 0.1, x + rw / 2 + 0.1, y - rd / 2 - 0.1, y + rd / 2 + 0.1)


def fill_shelves(sc, racks):
    rng = sc.rng
    rw, rd, _ = INFO["worn_metal_rack"]["size"]
    shelf_items = ["cardbox_A", "wooden_box", "plastic_container"]
    for rx, ry in racks:
        if rng.random() > 0.4:             # many racks stay empty (balance)
            continue
        for lvl, z in enumerate(SHELF_Z):
            if lvl + 1 < len(SHELF_Z):
                gap = SHELF_Z[lvl + 1] - SHELF_THICKNESS - z - 0.01
            else:
                gap = 0.6                               # on top of the rack
            if rng.random() < 0.25:
                continue
            x_left = rx - rw / 2 + SHELF_X_MARGIN
            x_right = rx + rw / 2 - SHELF_X_MARGIN
            depth = rd - 2 * SHELF_Y_MARGIN
            x = x_left
            while True:
                fits = []
                for n in shelf_items:
                    w, d, h = INFO[n]["size"]
                    for yaw, fw, fd in ((0, w, d), (90, d, w)):
                        if h <= gap and fd <= depth and x + fw <= x_right:
                            fits.append((n, yaw, fw, fd))
                if not fits or rng.random() < 0.2:
                    break
                n, yaw, fw, fd = rng.choice(fits)
                slack = (depth - fd) / 2
                sc.place(n, x + fw / 2, ry + rng.uniform(-slack, slack), z + 0.002,
                         yaw + rng.uniform(-2, 2))
                x += fw + rng.uniform(0.02, 0.15)


def load_pallet(sc, px, py, yaw):
    rng = sc.rng
    pw, pd, ph = INFO["pallet2v1_LOD0"]["size"]
    kind = rng.choices(["cardbox_A", "SM_Crate_Big", "plastic_container", "wooden_box", None],
                       weights=[3, 3, 3, 1, 2])[0]
    if kind is None:
        return
    look = sc.pick_look(kind)            # one look per load, like real shipments
    if kind == "SM_Crate_Big":
        sc.place(kind, px, py, ph, yaw + rng.uniform(-3, 3), look)
        return
    w, d, h = INFO[kind]["size"]
    best = max(((int(pw * 1.04 // w), int(pd * 1.04 // d), 0, w, d),
                (int(pw * 1.04 // d), int(pd * 1.04 // w), 90, d, w)),
               key=lambda o: o[0] * o[1])
    nx, ny, extra, fw, fd = best
    nx, ny = max(nx, 1), max(ny, 1)
    layers = rng.randint(1, {"cardbox_A": 3, "plastic_container": 3, "wooden_box": 3}[kind])
    t = math.radians(yaw)
    for layer in range(layers):
        for i in range(nx):
            for j in range(ny):
                if layer > 0 and rng.random() < 0.2:
                    continue
                lx = (i - (nx - 1) / 2) * fw + rng.uniform(-0.02, 0.02)
                ly = (j - (ny - 1) / 2) * fd + rng.uniform(-0.02, 0.02)
                wx = px + lx * math.cos(t) - ly * math.sin(t)
                wy = py + lx * math.sin(t) + ly * math.cos(t)
                sc.place(kind, wx, wy, ph + layer * h, yaw + extra + rng.uniform(-3, 3), look)


def pallet_stack(sc):
    """A stack of empty pallets, as seen in real warehouses."""
    rng = sc.rng
    spot = sc.find_floor_spot("pallet2v1_LOD0")
    if not spot:
        return
    x, y, yaw = spot
    ph = INFO["pallet2v1_LOD0"]["size"][2]
    look = sc.pick_look("pallet2v1_LOD0")
    for k in range(rng.randint(2, 10)):
        sc.place("pallet2v1_LOD0", x + rng.uniform(-0.03, 0.03), y + rng.uniform(-0.03, 0.03),
                 k * ph, yaw + rng.uniform(-3, 3), look)


# ---------------- distractors (unlabeled) ----------------
def add_shape(sc, kind, x, y, z_bottom, size, yaw):
    path = f"{PROPS}/shape_{sc.n:04d}"
    sc.n += 1
    sx, sy, sz = size
    if kind == "cube":
        g = UsdGeom.Cube.Define(stage, path)
        g.CreateSizeAttr(1.0)
        center_z = z_bottom + sz / 2
    elif kind == "cylinder":
        g = UsdGeom.Cylinder.Define(stage, path)
        g.CreateRadiusAttr(sx / 2)
        g.CreateHeightAttr(sz)
        g.CreateAxisAttr("Z")
        sx = sy = sz = 1.0
        center_z = z_bottom + size[2] / 2
    else:
        g = UsdGeom.Sphere.Define(stage, path)
        g.CreateRadiusAttr(sx / 2)
        sx = sy = sz = 1.0
        center_z = z_bottom + size[0] / 2
    api = UsdGeom.XformCommonAPI(g)
    api.SetTranslate(Gf.Vec3d(x, y, center_z))
    api.SetRotate(Gf.Vec3f(0.0, 0.0, yaw))
    if kind == "cube":
        api.SetScale(Gf.Vec3f(sx, sy, sz))
    bind(g.GetPrim(), sc.rng.choice(DISTRACT_LOOKS))


def add_distractors(sc):
    rng = sc.rng
    for name, lo, hi in (("forklift", 0, 2), ("Barrel_01", 2, 6),
                         ("portable_generator", 0, 2)):
        for _ in range(rng.randint(lo, hi)):
            spot = sc.find_floor_spot(name, pad=0.3 if name == "forklift" else 0.05)
            if spot:
                sc.place(name, spot[0], spot[1], 0.0, spot[2])

    for _ in range(rng.randint(1, 4)):                 # cabinets / machines
        w, d, h = rng.uniform(0.6, 1.5), rng.uniform(0.4, 0.8), rng.uniform(1.0, 2.2)
        spot = sc.find_rect(w, d)
        if spot:
            add_shape(sc, "cube", spot[0], spot[1], 0.0, (w, d, h), spot[2])
    for _ in range(rng.randint(2, 6)):                 # drums, cable rolls, pipes
        r2, h = rng.uniform(0.3, 1.0), rng.uniform(0.3, 1.5)
        spot = sc.find_rect(r2, r2)
        if spot:
            add_shape(sc, "cylinder", spot[0], spot[1], 0.0, (r2, r2, h), 0.0)
    for _ in range(rng.randint(0, 2)):
        r2 = rng.uniform(0.3, 0.8)
        spot = sc.find_rect(r2, r2)
        if spot:
            add_shape(sc, "sphere", spot[0], spot[1], 0.0, (r2, r2, r2), 0.0)

    for _ in range(rng.randint(4, 12)):                # signs / posters on walls
        w, h = rng.uniform(0.4, 1.6), rng.uniform(0.3, 1.0)
        z = rng.uniform(1.2, 4.5)
        side = rng.choice("SNEW")
        if side in "SN":
            x = rng.uniform(1.5, W - 1.5)
            y = 0.35 if side == "S" else D - 0.35
            add_shape(sc, "cube", x, y, z, (w, 0.03, h), 0.0)
        else:
            y = rng.uniform(1.5, D - 1.5)
            x = 0.35 if side == "W" else W - 0.35
            add_shape(sc, "cube", x, y, z, (0.03, w, h), 0.0)


def build_props(rng):
    if stage.GetPrimAtPath(PROPS):
        stage.RemovePrim(PROPS)
    UsdGeom.Xform.Define(stage, PROPS)
    sc = Scene(rng)
    racks = read_racks()
    add_racks_to_occupancy(sc, racks)
    fill_shelves(sc, racks)

    for _ in range(rng.randint(1, 4)):                        # empty pallet stacks
        pallet_stack(sc)
    for _ in range(rng.randint(8, 16)):                       # pallets with loads
        spot = sc.find_floor_spot("pallet2v1_LOD0")
        if spot:
            x, y, yaw = spot
            sc.place("pallet2v1_LOD0", x, y, 0.0, yaw)
            if rng.random() < 0.85:
                load_pallet(sc, x, y, yaw)

    for _ in range(rng.randint(6, 14)):                       # floor stacks
        name = rng.choices(["cardbox_A", "plastic_container", "wooden_box", "SM_Crate_Big"],
                           weights=[2, 3, 1, 4])[0]
        spot = sc.find_floor_spot(name)
        if not spot:
            continue
        x, y, yaw = spot
        h = INFO[name]["size"][2]
        look = sc.pick_look(name)
        for k in range(rng.randint(1, 2 if name == "SM_Crate_Big" else 3)):
            sc.place(name, x + rng.uniform(-0.03, 0.03), y + rng.uniform(-0.03, 0.03),
                     k * h, yaw + rng.uniform(-6, 6), look)

    add_distractors(sc)
    return sc


# ---------------- lighting ----------------
def randomize_lights(sc):
    rng = sc.rng
    root = stage.GetPrimAtPath(f"{SHELL}/Lights")
    if root:
        for p in root.GetChildren():
            if p.IsA(UsdLux.RectLight):
                light = UsdLux.RectLight(p)
                light.GetIntensityAttr().Set(rng.uniform(*CEILING_INTENSITY))
                light.GetColorTemperatureAttr().Set(rng.uniform(3800, 6500))
                img = UsdGeom.Imageable(p)
                img.MakeInvisible() if rng.random() < 0.1 else img.MakeVisible()
            elif p.IsA(UsdLux.DomeLight):
                UsdLux.DomeLight(p).GetIntensityAttr().Set(rng.uniform(*DOME_INTENSITY))
    # fill lights near objects, so shelves are not black holes
    for k in range(rng.randint(*FILL_LIGHTS)):
        if sc.targets:
            tx, ty, _, _ = rng.choice(sc.targets)
        else:
            tx, ty = rng.uniform(2, W - 2), rng.uniform(2, D - 2)
        light = UsdLux.SphereLight.Define(stage, f"{PROPS}/fill_{k}")
        light.CreateRadiusAttr(rng.uniform(0.2, 0.5))
        light.CreateIntensityAttr(rng.uniform(20000, 70000))
        light.CreateEnableColorTemperatureAttr(True)
        light.CreateColorTemperatureAttr(rng.uniform(3800, 6500))
        UsdGeom.XformCommonAPI(light.GetPrim()).SetTranslate(Gf.Vec3d(
            min(max(tx + rng.uniform(-1.5, 1.5), 0.5), W - 0.5),
            min(max(ty + rng.uniform(-1.5, 1.5), 0.5), D - 0.5),
            rng.uniform(2.0, 3.5)))


# ---------------- camera ----------------
if stage.GetPrimAtPath(CAM_PATH):
    stage.RemovePrim(CAM_PATH)
cam = UsdGeom.Camera.Define(stage, CAM_PATH)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
cam_op = UsdGeom.Xformable(cam).AddTransformOp()

CAM_SETTINGS = {           # height (m), distance to target (m), focal length (mm)
    "low":      dict(z=(0.4, 1.2), dist=(1.5, 9.0), focal=(10, 20)),
    "forklift": dict(z=(1.2, 2.5), dist=(2.0, 9.0), focal=(10, 22)),
    "drone":    dict(z=(4.0, H - 1.5), dist=(3.0, 10.0), focal=(14, 24)),
}


def set_camera(sc):
    rng = sc.rng
    mode = rng.choices(list(CAM_MODES), weights=list(CAM_MODES.values()))[0]
    cs = CAM_SETTINGS[mode]
    for _ in range(60):
        if sc.targets and rng.random() < 0.85:
            cls = rng.choice(sorted({t[3] for t in sc.targets}))
            tx, ty, tz, _ = rng.choice([t for t in sc.targets if t[3] == cls])
        else:
            tx, ty, tz = rng.uniform(3, W - 3), rng.uniform(3, D - 3), rng.uniform(0.3, 1.5)
        dist = rng.uniform(*cs["dist"])
        ang = rng.uniform(0, 2 * math.pi)
        ex, ey = tx + dist * math.cos(ang), ty + dist * math.sin(ang)
        ez = rng.uniform(*cs["z"])
        if not (0.5 < ex < W - 0.5 and 0.5 < ey < D - 0.5):
            continue
        if mode != "drone" and sc.occ.inside(ex, ey, margin=0.6):
            continue
        if mode == "low":            # look more level, like a robot or a hand-held phone
            tz = ez + rng.uniform(-0.6, 0.4)
        view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(ex, ey, ez), Gf.Vec3d(tx, ty, tz),
                                       Gf.Vec3d(0, 0, 1))
        cam_op.Set(view.GetInverse())
        cam.GetFocalLengthAttr().Set(rng.uniform(*cs["focal"]))
        return mode
    return None


# ---------------- outputs ----------------
img_dir = os.path.join(OUT_DIR, "images")
lbl_dir = os.path.join(OUT_DIR, "labels")
os.makedirs(img_dir, exist_ok=True)
os.makedirs(lbl_dir, exist_ok=True)


def to_yolo(bb, w, h):
    lines = []
    data, id2lab = bb["data"], bb["info"].get("idToLabels", {})
    for row in data:
        lab = id2lab.get(str(int(row["semanticId"])), {})
        name = lab.get("class") if isinstance(lab, dict) else lab
        name = (name or "").split(",")[0].strip()
        if name not in CLASSES:
            continue
        occ = float(row["occlusionRatio"]) if "occlusionRatio" in data.dtype.names else 0.0
        if occ == occ and occ > MAX_OCCLUSION:          # occ == occ skips NaN
            continue
        x0, y0 = max(0, row["x_min"]), max(0, row["y_min"])
        x1, y1 = min(w - 1, row["x_max"]), min(h - 1, row["y_max"])
        bw, bh = x1 - x0, y1 - y0
        if bw < MIN_BOX_PX or bh < MIN_BOX_PX:
            continue
        lines.append(f"{CLASSES.index(name)} {(x0 + x1) / 2 / w:.6f} "
                     f"{(y0 + y1) / 2 / h:.6f} {bw / w:.6f} {bh / h:.6f}")
    return lines


def write_meta(counts, n_images, modes):
    try:
        import omni.kit.app
        version = omni.kit.app.get_app().get_app_version()
    except Exception:
        version = "unknown"
    meta = dict(
        generator="v2", seed=SEED, num_scenes=NUM_SCENES, views_per_scene=VIEWS_PER_SCENE,
        resolution=RESOLUTION, rt_subframes=RT_SUBFRAMES, classes=CLASSES,
        shelf_z=SHELF_Z, shelf_thickness=SHELF_THICKNESS,
        max_occlusion=MAX_OCCLUSION, min_box_px=MIN_BOX_PX,
        warehouse_size=[W, D, H], assets=list(ASSETS), images=n_images,
        boxes_per_class=dict(zip(CLASSES, counts)), images_per_camera_mode=modes,
        appearance=dict(p_original=P_ORIGINAL_LOOK, tints=TINTS_PER_CLASS,
                        flats=FLATS_PER_CLASS, textures=len(TEXTURES)),
        lighting=dict(ceiling=CEILING_INTENSITY, dome=DOME_INTENSITY, fill=FILL_LIGHTS),
        camera_modes=CAM_MODES, walls=dict(p_keep_brick=P_KEEP_BRICK, colors=WALL_COLORS),
        isaac_sim=version,
        created=datetime.datetime.now().isoformat(timespec="seconds"),
    )
    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(OUT_DIR, "data.yaml"), "w") as f:
        f.write(f"path: {OUT_DIR}\ntrain: images\nval: images\n")
        f.write(f"nc: {len(CLASSES)}\nnames: {CLASSES}\n")


# ---------------- main loop ----------------
async def main():
    try:
        rep.orchestrator.set_capture_on_play(False)
    except Exception:
        pass
    rp = rep.create.render_product(CAM_PATH, RESOLUTION)
    rgb_a = rep.AnnotatorRegistry.get_annotator("rgb")
    bb_a = rep.AnnotatorRegistry.get_annotator(
        "bounding_box_2d_tight", init_params={"semanticTypes": ["class"]})
    rgb_a.attach([rp])
    bb_a.attach([rp])

    counts = [0] * len(CLASSES)
    modes = {m: 0 for m in CAM_MODES}
    saved = 0
    skipped = 0
    w, h = RESOLUTION
    await rep.orchestrator.step_async(rt_subframes=4)   # warm-up

    for s in range(NUM_SCENES):
        rng = random.Random(SEED * 1000 + s)           # each scene reproducible
        sc = build_props(rng)
        randomize_lights(sc)
        randomize_walls(rng)
        for v in range(VIEWS_PER_SCENE):
            mode = set_camera(sc)
            if mode is None:
                continue
            await rep.orchestrator.step_async(rt_subframes=RT_SUBFRAMES)
            rgb = rgb_a.get_data()
            if rgb is None or rgb.size == 0:
                continue
            arr = np.asarray(rgb)[:, :, :3]
            if arr.mean() < MIN_MEAN_BRIGHTNESS or arr.std() < MIN_STD:
                skipped += 1
                print(f"   skipped blank/black frame s{s:04d}_v{v:02d}")
                continue
            lines = to_yolo(bb_a.get_data(), w, h)
            stem = f"s{s:04d}_v{v:02d}_{mode}"
            Image.fromarray(arr).save(
                os.path.join(img_dir, stem + ".png"))
            with open(os.path.join(lbl_dir, stem + ".txt"), "w") as f:
                f.write("\n".join(lines))
            for ln in lines:
                counts[int(ln.split()[0])] += 1
            modes[mode] += 1
            saved += 1
        print(f"scene {s + 1}/{NUM_SCENES}: {sc.n} props, {saved} images so far")

    write_meta(counts, saved, modes)
    rgb_a.detach()
    bb_a.detach()
    rp.destroy()
    print(f"DONE: {saved} images in {OUT_DIR}")
    print("boxes per class:", dict(zip(CLASSES, counts)))
    print("images per camera mode:", modes)
    print(f"blank frames skipped: {skipped}")


asyncio.ensure_future(main())
