"""Zero-shot VLM check, arm B: WeVisOne-VL-2B (Qwen3-VL 2B + 9.6M pixel decoder), segmentation task.

Runs Tencent's inference package from the cloned folder (external/WeVisOne-VL-code, not pip-installed) in the
isolated venv external/wevisone_env. Class-agnostic query on the first 200 holdout strips (same as arm A);
"None" / no boxes = predicts clean. Scored against the union of all labelled classes.
"""
import json, os, sys, time
import numpy as np, pandas as pd, cv2
from PIL import Image

ROOT = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal"
sys.path.insert(0, os.path.join(ROOT, "external", "WeVisOne-VL-code"))
sys.path.insert(0, os.path.join(ROOT, "src"))
from train_lora import DATA_DIR, MASK_DIR, SPLIT_DIR  # noqa: E402
from wevisone_vl import WeVisOneVL  # noqa: E402

QUERY = "defect on the steel surface"
OUT = os.path.join(ROOT, "outputs", "vlm")


def dice(p, g):
    if not p.any() and not g.any():
        return 1.0
    return float(2 * (p & g).sum() / (p.sum() + g.sum()))


def main(n=200):
    # a CUDA OOM poisons the context for the rest of the process (seen 10-27 on strip 16): exit with code 3 without
    # writing the row, so the wrapper loop restarts and resumes; only successful rows count as done
    import torch
    out = os.path.join(OUT, "armB.jsonl")
    done = ({json.loads(l)["id"] for l in open(out) if "error" not in json.loads(l)} if os.path.exists(out) else set())
    ids = pd.read_csv(os.path.join(SPLIT_DIR, "holdout_ids.csv"))["ImageId"].tolist()[:n]
    if all(f in done for f in ids):
        print("B DONE nothing left", flush=True); return
    model = WeVisOneVL(os.path.join(ROOT, "external", "WeVisOne-VL-2B"), device="cuda", dtype="bfloat16",
                       decoder_batch_size=1)
    t0 = time.time(); k = 0
    with open(out, "a") as fo:
        for i, f in enumerate(ids):
            if f in done:
                continue
            g = cv2.imread(os.path.join(MASK_DIR, os.path.splitext(f)[0] + ".png"), cv2.IMREAD_UNCHANGED) > 0
            img = Image.open(os.path.join(DATA_DIR, "train_images", f)).convert("RGB")
            try:
                r = model.predict(img, task="segmentation", query=QUERY, temperature=0.0)
                pm = np.zeros(g.shape, bool)
                for m in r.masks:
                    m = np.asarray(m)
                    pm |= (m > (0.5 if m.dtype != bool else 0)) if m.shape == g.shape else False
                rec = {"id": f, "label_px": int(g.sum()), "n_boxes": len(r.boxes), "mask_px": int(pm.sum()),
                       "dice": dice(pm, g), "response": r.response[:300], "sec": round(r.latency_s, 2)}
            except Exception as e:
                if "out of memory" in repr(e).lower():
                    print(f"B OOM on {f} after {k} strips this process -> exit 3 for restart", flush=True)
                    sys.exit(3)
                rec = {"id": f, "label_px": int(g.sum()), "error": repr(e)[:300]}
            torch.cuda.empty_cache()
            fo.write(json.dumps(rec) + "\n"); fo.flush(); k += 1
            if k % 25 == 0 or k == 1:
                print(f"B {i + 1}/{len(ids)} | {(time.time() - t0) / k:.1f} s/strip | last {rec.get('response', rec.get('error'))!r:.120}", flush=True)
    print(f"B DONE {k} strips in {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
