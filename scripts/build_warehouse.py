"""
Build the warehouse shell in Isaac Sim from per-asset USD files.
Run in Isaac Sim: File -> New, then Window -> Script Editor -> paste -> Run.

Builds: floor, walls (with garage door, personnel door, windows), roof,
racks and ceiling lights. Same SEED = same warehouse.
"""
import math
import os
import random
import omni.usd
from pxr import Gf, Usd, UsdGeom, UsdLux

# ---------------- settings: edit these ----------------
ASSET_DIR = "C:/path/to/usd_assets"   # folder made by export_assets_usd.py
SEED = 42
TILE = 3.0          # module size (m)
COLS, ROWS = 10, 6  # 10 x 6 tiles = 30 m x 18 m
WALL_ROWS = 3       # 3 x 3 m = 9 m tall walls
ROOT = "/World/Warehouse"

# File names in ASSET_DIR (without .usdc). Match these to your files.
A = {
    "floor":       ["floor"],          # add more variants here later
    "roof":        "roof",
    "wall":        "wall_standard_standard_01",
    "window_wall": "wall_window_tall_large_01",
    "window":      "window_tall_large_01",
    "garage_wall": "wall_door_garage_door_01",
    "garage_door": "door_garage_door_01",
    "door_wall":   "wall_door_recessed_small_01",
    "door":        "door_recessed_small_01",
    "rack":        "worn_metal_rack",
}
# ------------------------------------------------------

W, D = COLS * TILE, ROWS * TILE
H = WALL_ROWS * TILE
rng = random.Random(SEED)
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)


def asset_path(name):
    return os.path.join(ASSET_DIR, name + ".usdc").replace("\\", "/")


# Stop early if any file is missing
names = A["floor"] + [v for k, v in A.items() if k != "floor"]
missing = [n for n in names if not os.path.exists(asset_path(n))]
if missing:
    raise FileNotFoundError(f"Missing in {ASSET_DIR}: {missing}")

if stage.GetPrimAtPath(ROOT):
    stage.RemovePrim(ROOT)
UsdGeom.Xform.Define(stage, ROOT)


def place(path, name, pos, rot_z=0.0):
    """Parent Xform holds our transform; child holds the asset reference."""
    parent = UsdGeom.Xform.Define(stage, path)
    api = UsdGeom.XformCommonAPI(parent)
    api.SetTranslate(Gf.Vec3d(*pos))
    api.SetRotate(Gf.Vec3f(0.0, 0.0, rot_z))
    child = stage.DefinePrim(path + "/asset")
    child.GetReferences().AddReference(asset_path(name))
    child.SetInstanceable(True)   # shared in GPU memory
    return parent


# ---------- floor ----------
for i in range(COLS):
    for j in range(ROWS):
        place(f"{ROOT}/Floor/tile_{i}_{j}", rng.choice(A["floor"]),
              (i * TILE, j * TILE, 0.0))

# ---------- roof ----------
for i in range(COLS):
    for j in range(ROWS):
        place(f"{ROOT}/Roof/tile_{i}_{j}", A["roof"], (i * TILE, j * TILE, H))


# ---------- walls ----------
# Facade pieces should span X from -width to 0, thickness toward +Y.
# Some pieces have a different origin, so each piece is measured once and
# shifted so it fills exactly its own slot in the wall.
_SHIFT = {}


def x_shift(name, span):
    key = (name, span)
    if key not in _SHIFT:
        probe = "/World/_Probe"
        UsdGeom.Xform.Define(stage, probe)
        stage.DefinePrim(probe + "/asset").GetReferences().AddReference(asset_path(name))
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                  [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        xmin = cache.ComputeWorldBound(stage.GetPrimAtPath(probe)).ComputeAlignedRange().GetMin()[0]
        stage.RemovePrim(probe)
        _SHIFT[key] = -span * TILE - xmin
        if abs(_SHIFT[key]) > 0.01:
            print(f"Aligning {name}: shifted {_SHIFT[key]:+.2f} m to fit its slot")
    return _SHIFT[key]


def wall_pos(side, k, span, z, name):
    if side == "S":
        (x, y), rot = ((k + span) * TILE, 0.0), 0.0
    elif side == "N":
        (x, y), rot = (k * TILE, D), 180.0
    elif side == "E":
        (x, y), rot = (W, (k + span) * TILE), 90.0
    else:                                   # "W"
        (x, y), rot = (0.0, k * TILE), -90.0
    dx = x_shift(name, span)
    t = math.radians(rot)
    return (x + dx * math.cos(t), y + dx * math.sin(t), z), rot


def build_side(side, modules):
    garage_at = 2 if side == "S" else None   # 6 m loading door, south wall
    door_at = 6 if side == "S" else None      # personnel door, south wall
    for row in range(WALL_ROWS):
        z = row * TILE
        k = 0
        while k < modules:
            base = f"{ROOT}/Walls/{side}_{row}_{k}"
            if row == 0 and k == garage_at:
                pos, rot = wall_pos(side, k, 2, z, A["garage_wall"])
                place(base, A["garage_wall"], pos, rot)
                place(base + "_door", A["garage_door"], pos, rot)
                k += 2
                continue
            if row == 0 and k == door_at:
                pos, rot = wall_pos(side, k, 1, z, A["door_wall"])
                place(base, A["door_wall"], pos, rot)
                place(base + "_door", A["door"], pos, rot)
            elif row == WALL_ROWS - 1 and k % 2 == 1:
                pos, rot = wall_pos(side, k, 1, z, A["window_wall"])
                place(base, A["window_wall"], pos, rot)
                place(base + "_glass", A["window"], pos, rot)
            else:
                pos, rot = wall_pos(side, k, 1, z, A["wall"])
                place(base, A["wall"], pos, rot)
            k += 1


if stage.GetPrimAtPath("/World/_Probe"):
    stage.RemovePrim("/World/_Probe")
build_side("S", COLS)
build_side("N", COLS)
build_side("E", ROWS)
build_side("W", ROWS)

# ---------- racks ----------
# Rows along X with about 3 m aisles; some gaps for variety.
RACK_W = 1.10
for r, y in enumerate([4.5, 8.5, 12.5]):
    x, n = 4.0, 0
    while x < W - 4.0:
        if rng.random() > 0.15:
            place(f"{ROOT}/Racks/row{r}_{n}", A["rack"], (x, y, 0.0),
                  0.0)
        x += RACK_W
        n += 1

# ---------- lights ----------
UsdGeom.Xform.Define(stage, f"{ROOT}/Lights")
n = 0
for x in range(3, int(W), 6):
    for y in range(3, int(D), 6):
        if rng.random() < 0.1:          # some bulbs off
            continue
        light = UsdLux.RectLight.Define(stage, f"{ROOT}/Lights/bay_{n}")
        light.CreateWidthAttr(2.0)
        light.CreateHeightAttr(0.6)
        light.CreateIntensityAttr(rng.uniform(80000, 150000))
        light.CreateEnableColorTemperatureAttr(True)
        light.CreateColorTemperatureAttr(rng.uniform(4000, 6500))
        UsdGeom.XformCommonAPI(light.GetPrim()).SetTranslate(
            Gf.Vec3d(x, y, H - 0.5))     # RectLight shines down (-Z)
        n += 1

dome = UsdLux.DomeLight.Define(stage, f"{ROOT}/Lights/sky")
dome.CreateIntensityAttr(500)

print(f"Warehouse built: {W} x {D} x {H} m, seed {SEED}, {n} ceiling lights")
