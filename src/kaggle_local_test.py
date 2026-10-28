"""Dry-run the Kaggle cells locally on outputs/kaggle_local_test (48 test images, the V79 bundles).

  python kaggle_local_test.py v1          the validated v1 cell, paths patched, V79 config, first 48 images
  python kaggle_local_test.py v2          the v2 cell, V79 config, two worker threads on the one local GPU
  python kaggle_local_test.py compare     v1 vs v2 submissions must be identical for the DINOv3-only V79
  python kaggle_local_test.py v2 <ckpt> [<ckpt> ...]   v2 on other bundles (e.g. adapter models)
"""
import os, sys, re
import pandas as pd

ROOT = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal"
T = os.path.join(ROOT, "outputs", "kaggle_local_test")
INPUT, N = os.path.join(T, "input").replace("\\", "/"), 48
V79 = ('CKPTS = ["model_vitb.pt", "model_vits_lora_r16.pt", "model_lora_r8.pt", "model_bce16.pt", '
       '"model_vitb_lora_warm_pseudo01.pt"]\nPIX = [0.65, 0.60, 0.65, 0.58]\nMINA = [630, 1470, 1050, 3150]\n')


def run_v1():
    work = os.path.join(T, "work_v1").replace("\\", "/")
    src = open(os.path.join(ROOT, "src", "kaggle_ensemble_cell.py"), encoding="utf-8").read()
    src = src.replace("/kaggle/input", INPUT).replace("/kaggle/working", work)
    src = re.sub(r"^CKPTS\s*=.*$", "", src, flags=re.M)
    src = re.sub(r"^PIX\s*=.*$", "", src, flags=re.M)
    src = re.sub(r"^MINA\s*=.*$", "", src, flags=re.M)
    src = src.replace("WEIGHTS   = None", V79 + "WEIGHTS   = None", 1)
    src = src.replace('print(f"columns {columns}', f"image_ids = image_ids[:{N}]\nprint(f\"columns {{columns}}", 1)
    src = src.replace('df.to_csv("submission.csv", index=False)', f'df.to_csv(r"{work}/submission.csv", index=False)')
    exec(compile(src, "kaggle_ensemble_cell_v1", "exec"), {"__name__": "__main__"})


def run_v2(ckpts=None):
    work = os.path.join(T, "work_v2").replace("\\", "/")
    os.environ.update({"KAGGLE_INPUT": INPUT, "KAGGLE_WORK": work, "LIMIT_IMAGES": str(N),
                       "DEVICES_OVERRIDE": os.environ.get("TEST_DEVICES", "cuda:0,cuda:0")})
    src = open(os.path.join(ROOT, "src", "kaggle_ensemble_cell_v2.py"), encoding="utf-8").read()
    if ckpts:
        src = re.sub(r'^CKPTS\s*=.*?(?=^WEIGHTS)', f"CKPTS = {ckpts!r}\n", src, count=1, flags=re.M | re.S)
    exec(compile(src, "kaggle_ensemble_cell_v2", "exec"), {"__name__": "__main__"})


def compare():
    a = pd.read_csv(os.path.join(T, "work_v1", "submission.csv")).fillna("")
    b = pd.read_csv(os.path.join(T, "work_v2", "submission.csv")).fillna("")
    same = (a.values == b.values).all(1)
    print(f"v1 rows {len(a)} | v2 rows {len(b)} | identical rows {same.sum()}/{len(a)} | "
          f"non-empty v1 {(a.EncodedPixels != '').sum()} v2 {(b.EncodedPixels != '').sum()}")
    if not same.all():
        print(a[~same].head(), b[~same].head(), sep="\n")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "v1":
        run_v1()
    elif cmd == "v2":
        run_v2(sys.argv[2:] or None)
    else:
        compare()
