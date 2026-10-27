"""Slide figures: raw / ground truth / prediction overlays from the best single model.

Model: soft ViT-B (kaggle_bundle_calib/model_vitb_lora_warm_soft_pseudo01.pt, Kaggle 0.90695),
Kaggle-cell inference (3-view TTA, fp16, bilinear), rate-matched thresholds [.51,.43,.44,.32],
min area [1050,2450,1750,5250].

  A  the same four strips as figures/zero_shot_dinov3_probe.png (all in train12k: SEEN images)
  B  four random holdout strips (seed 0), unseen, not curated
  C  curated holdout strips that illustrate: a clean hit per class, a mask-shape error,
     a false alarm, a missed defect. Selection rule is printed on each panel.
"""
import os, json
import numpy as np, pandas as pd, cv2, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from pseudo_teacher import load_member, member_probs
from train_lora import DATA_DIR, MASK_DIR, SPLIT_DIR

REL = os.path.join("kaggle_bundle_calib", "model_vitb_lora_warm_soft_pseudo01.pt")
PIX, MINA = [.51, .43, .44, .32], [1050, 2450, 1750, 5250]
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
COL = {1: (230, 159, 0), 2: (86, 180, 233), 3: (0, 158, 115), 4: (213, 94, 0)}      # Okabe-Ito, colour-blind safe
OUT = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\figures\slides"
PCA_STRIPS = ["0002cc93b", "002fc4e19", "0007a71bf", "000f6bf48"]


def read(i):
    return cv2.imread(os.path.join(DATA_DIR, "train_images", i + ".jpg"))


def gt_mask(i):
    return cv2.imread(os.path.join(MASK_DIR, i + ".png"), cv2.IMREAD_UNCHANGED)


@torch.no_grad()
def predict(net, ids, bs=4):
    """-> dict id -> (256,1600) uint8 class map (0 bg, 1-4), after the submission decision rule."""
    res = {}
    for i0 in range(0, len(ids), bs):
        chunk = ids[i0:i0 + bs]
        x = torch.from_numpy(np.stack([((read(i)[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD)
                                       .transpose(2, 0, 1) for i in chunk])).cuda()
        p = member_probs(net, x, ("none", "h", "v"), True)
        for k, i in enumerate(chunk):
            cm = np.zeros((256, 1600), np.uint8)
            for j in range(4):
                full = cv2.resize(p[k, j + 1], (1600, 256), interpolation=cv2.INTER_LINEAR)
                m = full > PIX[j]
                if m.sum() >= MINA[j]:
                    cm[m & (cm == 0)] = j + 1
            res[i] = cm
        if (i0 // bs) % 25 == 0:
            print(f"  predicted {min(i0 + bs, len(ids))}/{len(ids)}", flush=True)
    return res


def pair_dice(pred, gt):
    d = np.zeros(4)
    for j in range(4):
        p, g = pred == j + 1, gt == j + 1
        s = p.sum() + g.sum()
        d[j] = 1.0 if s == 0 else 2 * (p & g).sum() / s
    return d


def overlay(img_bgr, cm, alpha=0.55):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    out = g.copy()
    for c, rgb in COL.items():
        m = cm == c
        out[m] = (1 - alpha) * g[m] + alpha * np.array(rgb, np.float32)
    return out.astype(np.uint8)


def panel(rows, path, title):
    """rows: list of (id, note, pred_map). Each row: raw / ground truth / prediction, stacked."""
    n = len(rows)
    fig, axs = plt.subplots(3 * n, 1, figsize=(16, 2.35 * 3 * n), dpi=110)
    for r, (i, note, pm) in enumerate(rows):
        img, gt = read(i), gt_mask(i)
        d = pair_dice(pm, gt)
        pres_g = [j + 1 for j in range(4) if (gt == j + 1).any()]
        pres_p = [j + 1 for j in range(4) if (pm == j + 1).any()]
        for k, (im, lab) in enumerate(((cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "raw"),
                                       (overlay(img, gt), "ground truth"),
                                       (overlay(img, pm), "model prediction"))):
            ax = axs[3 * r + k]
            ax.imshow(im, aspect="auto"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=10)
            for s in ax.spines.values():
                s.set_visible(False)
        axs[3 * r].set_title(f"{i}  |  {note}  |  GT classes {pres_g or 'none'}, predicted {pres_p or 'none'}"
                             + ("  |  Dice " + ", ".join(f"c{j + 1} {d[j]:.2f}" for j in range(4) if j + 1 in set(pres_g) | set(pres_p))
                                if (pres_g or pres_p) else ""), fontsize=10, loc="left")
    fig.suptitle(title, fontsize=13, y=0.999)
    fig.legend(handles=[Patch(color=np.array(COL[c]) / 255, label=f"defect {c}") for c in COL],
               loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 0.985))
    fig.savefig(path); plt.close(fig)
    print("saved", path, flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    net = load_member(REL, "cuda")
    hold = [os.path.splitext(f)[0] for f in pd.read_csv(os.path.join(SPLIT_DIR, "holdout_ids.csv"))["ImageId"]]

    print("A: same strips as the PCA figure (train12k, seen)", flush=True)
    pa = predict(net, PCA_STRIPS)
    panel([(i, f"class {k + 1} example, TRAINING image", pa[i]) for k, i in enumerate(PCA_STRIPS)],
          os.path.join(OUT, "A_same_strips_train.png"),
          "Soft ViT-B (Kaggle 0.90695) on the four strips of the frozen-DINO figure. Training images, so optimistic.")

    print("holdout inference", flush=True)
    ph = predict(net, hold)
    rec = []
    for i in hold:
        gt = gt_mask(i); d = pair_dice(ph[i], gt)
        rec.append({"id": i, **{f"gt{j + 1}": int((gt == j + 1).sum()) for j in range(4)},
                    **{f"pr{j + 1}": int((ph[i] == j + 1).sum()) for j in range(4)},
                    **{f"dice{j + 1}": float(d[j]) for j in range(4)}})
    df = pd.DataFrame(rec); df.to_csv(os.path.join(OUT, "holdout_pair_dice.csv"), index=False)

    rng = np.random.default_rng(0)
    rid = list(rng.choice(hold, 4, replace=False))
    panel([(i, "random holdout strip (seed 0)", ph[i]) for i in rid], os.path.join(OUT, "B_random_holdout.png"),
          "Four random holdout strips (unseen, seed 0, not curated)")

    sel = []
    for c in (1, 2, 3, 4):                                         # clean hit: defect present, Dice >= 0.8, median-sized
        t = df[(df[f"gt{c}"] > 0) & (df[f"dice{c}"] >= 0.8)].sort_values(f"gt{c}")
        if len(t):
            r = t.iloc[len(t) // 2]
            sel.append((r["id"], f"clean hit, class {c} (Dice >= 0.8, median-sized of {len(t)})"))
    t = df[(df.gt3 > 0) & (df.dice3 > 0.4) & (df.dice3 < 0.7) & (df.pr3 > 1.4 * df.gt3)]
    if len(t):
        sel.append((t.sort_values("pr3").iloc[len(t) // 2]["id"], "mask-shape error: class 3 detected, predicted area > 1.4x labelled"))
    fa = df[(((df.gt1 == 0) & (df.pr1 > 0)) | ((df.gt2 == 0) & (df.pr2 > 0)) | ((df.gt3 == 0) & (df.pr3 > 0)) | ((df.gt4 == 0) & (df.pr4 > 0)))]
    if len(fa):
        sel.append((fa.iloc[0]["id"], f"false alarm: first of {len(fa)} holdout strips with a fired class that has no label"))
    ms = df[(((df.gt1 > 0) & (df.pr1 == 0)) | ((df.gt2 > 0) & (df.pr2 == 0)) | ((df.gt3 > 0) & (df.pr3 == 0)) | ((df.gt4 > 0) & (df.pr4 == 0)))]
    if len(ms):
        sel.append((ms.iloc[0]["id"], f"missed defect: first of {len(ms)} holdout strips with a labelled class not predicted"))
    for g in range(0, len(sel), 4):
        panel([(i, n, ph[i]) for i, n in sel[g:g + 4]], os.path.join(OUT, f"C_curated_holdout_{g // 4 + 1}.png"),
              "Curated holdout strips (unseen). Chosen to illustrate, not representative.")
    json.dump({"random": [str(i) for i in rid], "curated": [[str(i), n] for i, n in sel],
               "n_false_alarm_strips": int(len(fa)), "n_missed_strips": int(len(ms))},
              open(os.path.join(OUT, "selection.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
