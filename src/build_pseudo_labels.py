"""Turn pseudo_teacher.py arrays into pseudo-labels - or, by default, only REPORT
what each rule would keep, so the go/no-go on the 8-hour student is made on the
real test-set numbers instead of the holdout estimate.

Rule per (image, class), derived on holdout:
  defect  all 5 members fire AND the ensemble fires          mask = avg prob > PIX
  clean   all 5 members empty AND ensemble max-prob < clean-max
  else    no label
On holdout the max-prob cut removed 17 of the 23 unanimous-empty misses (the
submission min_area suppresses real defects; it is tuned to score, not to be
true). Nothing removes the confident c3 false positives - they sit at 0.99-1.00.

An image is kept only if all 4 classes are labelled: the student uses the
softmax recipe, whose classes share one normalisation, so one class of one image
cannot be ignored on its own.

--write T  writes data/processed/pseudo_masks/<id>.png (same uint8 class-id
           format as the real masks) and data/processed/pseudo_ids.csv using
           clean-max T. Refuses to replace an existing pseudo_ids.csv without
           --overwrite.
"""
import argparse, json, os
import numpy as np, pandas as pd, cv2

ROOT = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal"
PSEUDO = os.path.join(ROOT, "outputs", "pseudo")
PROC = os.path.join(ROOT, "data", "processed")
PIX = [0.65, 0.60, 0.65, 0.58]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-max", type=float, nargs="+", default=[1.01, 0.2, 0.1, 0.05])
    ap.add_argument("--write", type=float, default=None, help="clean-max to write labels with")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--pseudo-dir", default=PSEUDO,
                    help="teacher output folder (pseudo_teacher.py or self_teacher.py layout)")
    ap.add_argument("--out-prefix", default="pseudo",
                    help="writes <prefix>_ids.csv and <prefix>_masks/ under data/processed")
    a = ap.parse_args()

    pdir = a.pseudo_dir
    ids = pd.read_csv(os.path.join(pdir, "test_ids.csv"))["ImageId"].tolist()
    prog = json.load(open(os.path.join(pdir, "progress.json")))
    assert prog["n"] == len(ids) and prog["next"] >= prog["n"], \
        f"teacher incomplete: {prog['next']}/{prog['n']}"
    pix = prog["pix"]                     # the mask threshold is the rule the teacher decided with
    assert len(pix) == 4, pix
    print(f"teacher {pdir} | voters {prog.get('voters', 'members')} | pix {pix}")
    n = len(ids)
    probs = np.load(os.path.join(pdir, "probs.npy"), mmap_mode="r")    # (n, 4, 64, 400)
    md = np.load(os.path.join(pdir, "member_dec.npy"))                 # (voters, n, 4)
    ens = np.load(os.path.join(pdir, "ens_dec.npy"))                   # (n, 4)
    mx = np.concatenate([np.asarray(probs[i:i + 500], np.float32).reshape(-1, 4, 64 * 400).max(-1)
                         for i in range(0, n, 500)])     # bilinear never exceeds the grid max

    defect = md.all(0) & ens
    unan_empty = (~md).all(0)
    print(f"{n} test images | {md.shape[0]} members | submission positive rate % "
          f"{(100 * ens.mean(0)).round(2).tolist()}")
    print(f"defect labels per class {defect.sum(0).tolist()} "
          f"(members unanimous but ensemble silent: {(md.all(0) & ~ens).sum(0).tolist()})")
    print(f"members split per class {(~(md.all(0) | unan_empty)).sum(0).tolist()}\n")
    print(f"{'clean-max':>9}{'clean c1':>10}{'c2':>7}{'c3':>7}{'c4':>7}"
          f"{'images kept':>13}{'with defect':>13}   defect pairs kept c1..c4")
    for t in a.clean_max:
        clean = unan_empty & (mx < t)
        keep = (defect | clean).all(1)
        c = clean.sum(0)
        print(f"{'none' if t > 1 else t:>9}{c[0]:>10}{c[1]:>7}{c[2]:>7}{c[3]:>7}"
              f"{keep.sum():>13}{defect[keep].any(1).sum():>13}   {defect[keep].sum(0).tolist()}")

    if a.write is None:
        print("\nreport only - nothing written (pass --write T to create labels)")
        return
    t = a.write
    keep = np.where((defect | (unan_empty & (mx < t))).all(1))[0]
    out_csv = os.path.join(PROC, f"{a.out_prefix}_ids.csv")
    mdir = os.path.join(PROC, f"{a.out_prefix}_masks")
    assert a.overwrite or not os.path.exists(out_csv), f"{out_csv} exists - pass --overwrite"
    os.makedirs(mdir, exist_ok=True)
    for i in keep:
        p = np.asarray(probs[i], np.float32)
        full = np.stack([cv2.resize(p[j], (1600, 256), interpolation=cv2.INTER_LINEAR) for j in range(4)])
        on = np.stack([(full[j] > pix[j]) & defect[i, j] for j in range(4)])
        # overlapping classes resolved by the higher probability, as a single-channel mask must
        m = np.where(on.any(0), np.where(on, full, -1.0).argmax(0) + 1, 0).astype(np.uint8)
        cv2.imwrite(os.path.join(mdir, os.path.splitext(ids[i])[0] + ".png"), m)
    pd.DataFrame({"ImageId": [ids[i] for i in keep]}).to_csv(out_csv, index=False)
    print(f"\nwrote {len(keep)} masks -> {mdir}\n      ids -> {out_csv}   (clean-max {t})")


if __name__ == "__main__":
    main()
