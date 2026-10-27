"""Self-teacher for per-member self-training (E22).

E21 showed that students sharing ONE teacher (the 0.90756 ensemble) converge: the
two pseudo ViT-S models disagree 29% less than their originals, and every extra
shared-teacher member lowered the ensemble's private score. Here a member labels
the test images for its OWN student, so the student inherits that member's
decision pattern instead of the ensemble's.

It is also the control E19 lacked: a solo gain here cannot come from distilling the
ensemble, only from training on test-distribution images.

The rule mirrors the ensemble rule with the model's own 3 TTA views as the voters
(a single model has no members to agree): defect = all 3 views fire AND the
TTA-averaged prediction fires; clean = all 3 views empty AND TTA-averaged max-prob
< clean-max. build_pseudo_labels.py applies it unchanged. Thresholds are the
single-model set - the rule this model is scored under on its own.

Runs the same rule on the 500 labelled holdout images first, for comparison with
the ensemble rule (E19 at clean-max 0.1: 433 images kept, 424 fully correct;
1,922 image-class pairs labelled, 14 wrong). Then writes the pseudo_teacher.py
layout to outputs/<out>/ with member_dec holding per-VIEW decisions.
"""
import argparse, json, os, time
import numpy as np, pandas as pd, cv2

import pseudo_teacher as PT
from pseudo_diag import load
from train_lora import DATA_DIR, MASK_DIR, SPLIT_DIR

PIX, MINA = PT.E17_PIX, PT.E17_MINA


def views(net, x):
    return [PT.member_probs(net, x, (v,), half=True) for v in PT.TTA]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="kaggle_bundle_lora/model_vits_lora_r16.pt")
    ap.add_argument("--out", default="pseudo_self_vits")
    ap.add_argument("--clean-max", type=float, default=0.1)
    a = ap.parse_args()
    device = "cuda"
    od = os.path.join(PT.OUTPUTS, a.out)
    assert not os.path.exists(os.path.join(od, "progress.json")), f"{od} already holds teacher output"
    net = PT.load_member(a.model, device)

    # ---- holdout: label quality of the self rule ----
    ids = pd.read_csv(os.path.join(SPLIT_DIR, "holdout_ids.csv"))["ImageId"].tolist()
    n = len(ids)
    vdec = np.zeros((3, n, 4), bool); edec = np.zeros((n, 4), bool)
    mx = np.zeros((n, 4), np.float32); gt = np.zeros((n, 4), bool)
    for i0 in range(0, n, 4):
        ch = ids[i0:i0 + 4]; sl = slice(i0, i0 + len(ch))
        vp = views(net, load(os.path.join(DATA_DIR, "train_images"), ch).to(device))
        for v in range(3):
            vdec[v, sl] = PT.decide(vp[v], PIX, MINA)[0]
        edec[sl], mx[sl] = PT.decide(sum(vp) / 3, PIX, MINA)
        for k, f in enumerate(ch):
            g = cv2.imread(os.path.join(MASK_DIR, os.path.splitext(f)[0] + ".png"), cv2.IMREAD_UNCHANGED)
            gt[i0 + k] = [(g == c).any() for c in (1, 2, 3, 4)]
    defect = vdec.all(0) & edec
    clean = (~vdec).all(0) & (mx < a.clean_max)
    keep = (defect | clean).all(1)
    wrong = (defect & ~gt) | (clean & gt)
    print(f"HOLDOUT self-rule ({os.path.basename(a.model)}, 3 TTA views as voters, clean-max {a.clean_max})")
    print(f"{'cls':>4}{'GT+':>6}{'defect ok/total':>17}{'clean ok/total':>16}{'unlabelled':>12}")
    for j in range(4):
        print(f"{j + 1:>4}{gt[:, j].sum():>6}{f'{(defect[:, j] & gt[:, j]).sum()}/{defect[:, j].sum()}':>17}"
              f"{f'{(clean[:, j] & ~gt[:, j]).sum()}/{clean[:, j].sum()}':>16}{(~(defect | clean))[:, j].sum():>12}")
    lab = int((defect | clean).sum())
    print(f"  pairs labelled {lab}, wrong {int(wrong.sum())} ({100 * wrong.sum() / max(lab, 1):.2f}%) | images kept "
          f"{int(keep.sum())}, fully correct {int((keep & ~wrong.any(1)).sum())}")
    print(f"  ensemble rule, same holdout (E19): pairs 1922, wrong 14 (0.73%) | images kept 433, fully correct 424", flush=True)

    # ---- test: write teacher arrays ----
    files = sorted(os.listdir(os.path.join(DATA_DIR, "test_images"))); nt = len(files)
    os.makedirs(od, exist_ok=True)
    probs = np.lib.format.open_memmap(os.path.join(od, "probs.npy"), mode="w+", dtype=np.float16, shape=(nt, 4, 64, 400))
    ens = np.zeros((nt, 4), bool); md = np.zeros((3, nt, 4), bool); mm = np.zeros((3, nt, 4), np.float16)
    t0 = time.time()
    for i0 in range(0, nt, 4):
        ch = files[i0:i0 + 4]; sl = slice(i0, i0 + len(ch))
        vp = views(net, load(os.path.join(DATA_DIR, "test_images"), ch).to(device))
        for v in range(3):
            md[v, sl], mm[v, sl] = PT.decide(vp[v], PIX, MINA)
        avg = sum(vp) / 3
        probs[sl] = avg[:, 1:].astype(np.float16)
        ens[sl] = PT.decide(avg, PIX, MINA)[0]
        if (i0 // 4) % 250 == 0:
            done = i0 + len(ch); el = time.time() - t0
            print(f"  test {done}/{nt} | ETA {(nt - done) * el / done / 60:5.1f} min", flush=True)
    probs.flush()
    np.save(os.path.join(od, "ens_dec.npy"), ens); np.save(os.path.join(od, "member_dec.npy"), md)
    np.save(os.path.join(od, "member_max.npy"), mm)
    pd.DataFrame({"ImageId": files}).to_csv(os.path.join(od, "test_ids.csv"), index=False)
    json.dump({"members": [f"{a.model} views {list(PT.TTA)}"], "voters": "tta_views", "tta": list(PT.TTA),
               "pix": PIX, "mina": MINA, "half": 1, "n": nt, "next": nt}, open(os.path.join(od, "progress.json"), "w"))
    print(f"\ntest positive rate % {(100 * ens.mean(0)).round(2).tolist()} | images with all classes unanimous across "
          f"views {int((md.all(0) | (~md).all(0)).all(1).sum())}/{nt} | {(time.time() - t0) / 60:.1f} min -> {od}")


if __name__ == "__main__":
    main()
