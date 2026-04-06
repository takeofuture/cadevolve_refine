import pickle
from pathlib import Path

# ========= 設定 =========
BASE_DIR = Path("/opt/CADEvolve")
PY_DIR = BASE_DIR / "dataset_utils" / "results" / "canonicalized_flat"
STL_DIR = BASE_DIR / "dataset_utils" / "results" / "rotated_stl"
OUT_DIR = BASE_DIR / "data" / "train_set"

OUT_PKL = OUT_DIR / "pairs.pkl"
OUT_MISSING = OUT_DIR / "missing_pairs.txt"

# ========= チェック =========
if not PY_DIR.exists():
    raise FileNotFoundError(f"PY_DIR not found: {PY_DIR}")

if not STL_DIR.exists():
    raise FileNotFoundError(f"STL_DIR not found: {STL_DIR}")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========= .py を辞書化 =========
# key: ファイル名そのもの
# val: 絶対パス
py_map = {}
duplicate_py_names = []

for py_path in sorted(PY_DIR.glob("*.py")):
    key = py_path.name
    if key in py_map:
        duplicate_py_names.append(key)
    py_map[key] = str(py_path.resolve())

if duplicate_py_names:
    print("WARNING: duplicated .py names found:")
    for name in duplicate_py_names[:20]:
        print("  ", name)

# ========= pairs 作成 =========
pairs = []
missing = []
bad_name = []

for stl_path in sorted(STL_DIR.glob("*.stl")):
    stl_name = stl_path.name

    # 例:
    # Z0_Y180_Z0__cone_revolve__08_standardized_centered_scaled_binarized.stl
    # -> cone_revolve__08_standardized_centered_scaled_binarized.py
    parts = stl_name.split("__", 1)

    if len(parts) != 2:
        bad_name.append(stl_name)
        continue

    py_name = parts[1].replace(".stl", ".py")
    py_path = py_map.get(py_name)

    if py_path is None:
        missing.append((stl_name, py_name))
        continue

    pairs.append((py_path, str(stl_path.resolve())))

# ========= 保存 =========
with open(OUT_PKL, "wb") as f:
    pickle.dump(pairs, f)

with open(OUT_MISSING, "w", encoding="utf-8") as f:
    if bad_name:
        f.write("[BAD STL NAME FORMAT]\n")
        for name in bad_name:
            f.write(f"{name}\n")
        f.write("\n")

    if missing:
        f.write("[MISSING MATCHING PY]\n")
        for stl_name, py_name in missing:
            f.write(f"STL: {stl_name}\n")
            f.write(f"PY : {py_name}\n\n")

# ========= レポート =========
print("===================================")
print(f"py files        : {len(py_map)}")
print(f"stl files       : {len(list(STL_DIR.glob('*.stl')))}")
print(f"pairs generated : {len(pairs)}")
print(f"bad stl names   : {len(bad_name)}")
print(f"missing matches : {len(missing)}")
print(f"pairs.pkl       : {OUT_PKL}")
print(f"missing report  : {OUT_MISSING}")
print("===================================")

if pairs:
    print("First 5 pairs:")
    for item in pairs[:5]:
        print(item)
