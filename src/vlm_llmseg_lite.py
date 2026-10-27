"""Zero-shot VLM check, arm C: an LLMSeg-style loop (arXiv 2610.29409) with local qwen3-vl:4b-instruct.

Own minimal implementation of the paper's method, not its code:
  1 detect   boxes of every surface defect (0-1000 grid, x first)
  2 segment  one closed outline per box (control points inside the box) -> periodic cubic B-spline -> mask
  3 refine   up to T passes: show the model an overlay of its current mask; it proposes signed focus boxes
             (add / remove); one more call traces an outline in each focus box; apply; stop when none proposed
Class-agnostic (our classes are unnamed): scored against the union of all labelled classes.
Results: outputs/vlm/armC/<id>.png (final mask) + armC.jsonl. Run after arm A (Ollama reloads on num_ctx change).
"""
import argparse, base64, json, os, re, time, urllib.request
import numpy as np, pandas as pd, cv2
from scipy.interpolate import splprep, splev

from train_lora import DATA_DIR, MASK_DIR, SPLIT_DIR

OUT = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\vlm"
URL = "http://localhost:11434/api/chat"
W, H, GRID = 1600, 256, 1000.0
QUERY = "surface defects on the steel sheet (scratches, pits, dents, stains, scale, cracks or other damage)"
COORD = ("The image is 1600 pixels wide and 256 pixels tall, but ALL coordinates you output use a NORMALIZED "
         "0-1000 grid on BOTH axes: x=0 left edge, x=1000 right edge, y=0 top edge, y=1000 bottom edge. "
         "Boxes are [x1, y1, x2, y2] with x1 < x2 and y1 < y2. Points are [x, y].")


def ask(model, prompt, img):
    body = {"model": model, "stream": False, "options": {"num_ctx": 8192, "temperature": 0, "num_predict": 1500},
            "messages": [{"role": "user", "content": prompt,
                          "images": [base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode()]}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=900).read())["message"]["content"]


def jload(s):
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def to_px(pts):
    p = np.asarray(pts, np.float64)
    return np.stack([p[:, 0] / GRID * W, p[:, 1] / GRID * H], 1)


def clip_box(b):
    x1, y1, x2, y2 = [float(v) for v in b]
    x1, x2 = sorted((max(0, min(GRID, x1)), max(0, min(GRID, x2))))
    y1, y2 = sorted((max(0, min(GRID, y1)), max(0, min(GRID, y2))))
    return [x1, y1, x2, y2] if x2 - x1 >= 2 and y2 - y1 >= 2 else None


def outline_mask(pts, box):
    """Closed periodic cubic B-spline through the control points, confined to the box."""
    pts = [p for p in pts if isinstance(p, (list, tuple)) and len(p) == 2]
    m = np.zeros((H, W), np.uint8)
    if len(pts) < 3:
        return m.astype(bool)
    xy = to_px(pts)
    if len(xy) >= 4:
        try:
            tck, _ = splprep([np.r_[xy[:, 0], xy[0, 0]], np.r_[xy[:, 1], xy[0, 1]]], s=0, per=1, k=3)
            xs, ys = splev(np.linspace(0, 1, 400), tck); xy = np.stack([xs, ys], 1)
        except Exception:
            pass
    cv2.fillPoly(m, [np.round(xy).astype(np.int32)], 1)
    bx = to_px([[box[0], box[1]], [box[2], box[3]]])
    conf = np.zeros_like(m)
    conf[int(bx[0, 1]):int(np.ceil(bx[1, 1])) + 1, int(bx[0, 0]):int(np.ceil(bx[1, 0])) + 1] = 1
    return (m & conf).astype(bool)


def overlay(gray, mask, boxes):
    o = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    o[mask] = 0.65 * o[mask] + 0.35 * np.array([20, 230, 70], np.float32)
    o = o.astype(np.uint8)
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(o, cs, -1, (0, 0, 255), 1)
    for b in boxes:
        p = np.round(to_px([[b[0], b[1]], [b[2], b[3]]])).astype(int)
        cv2.rectangle(o, tuple(p[0]), tuple(p[1]), (0, 255, 255), 1)
    return o


def trace(model, img, boxes, what):
    blk = "\n".join(f"  id {i}: box {[round(v) for v in b]}" for i, b in enumerate(boxes))
    r = jload(ask(model, f"{COORD}\nRegions ({what}):\n{blk}\nFor EACH region id, trace the outline of the "
                  f"{QUERY.split(' (')[0]} inside that box with 8 to 24 control points in order around it; every point "
                  "must lie inside that id's box. If a box holds no defect, return an empty list for it.\n"
                  'Return ONLY JSON: {"instances": [{"id": 0, "points": [[x, y], ...]}]}', img))
    out = {}
    for it in r.get("instances", []) if isinstance(r.get("instances"), list) else []:
        if isinstance(it, dict) and isinstance(it.get("id"), int) and 0 <= it["id"] < len(boxes):
            out[it["id"]] = it.get("points") or []
    return out


def run_one(model, gray, T):
    calls = 1
    d = jload(ask(model, f"{COORD}\nFind every distinct instance of: {QUERY}. Make each box generous enough to "
                  'contain the whole defect.\nReturn ONLY JSON: {"detections": [{"id": 0, "box": [x1, y1, x2, y2], '
                  '"confidence": 0.0}]}. If there is none, return {"detections": []}.', gray))
    boxes = [b for b in (clip_box(x.get("box", [])) if isinstance(x, dict) and isinstance(x.get("box"), list)
                         and len(x["box"]) == 4 else None for x in d.get("detections", []) or []) if b][:6]
    mask = np.zeros((H, W), bool)
    if not boxes:
        return mask, mask.copy(), {"boxes": [], "calls": calls, "passes": 0}
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    pts = trace(model, rgb, boxes, "detected defects"); calls += 1
    for i, b in enumerate(boxes):
        mask |= outline_mask(pts.get(i, []), b)
    seg = mask.copy(); passes = 0
    for t in range(T):
        f = jload(ask(model, f"{COORD}\nThe image shows the CURRENT segmentation of the {QUERY.split(' (')[0]}: "
                      "green tint = current mask, red line = its boundary, yellow rectangles = detection boxes. "
                      "Compare the mask with the real defects underneath. Propose up to 6 focus boxes where the mask "
                      'is wrong: "add" where a defect is not covered, "remove" where clean steel is covered.\n'
                      'Return ONLY JSON: {"focus": [{"box": [x1, y1, x2, y2], "op": "add"}]}; return {"focus": []} '
                      "if the mask is already right.", overlay(gray, mask, boxes))); calls += 1
        fb = [(clip_box(x["box"]), x.get("op")) for x in f.get("focus", []) or []
              if isinstance(x, dict) and isinstance(x.get("box"), list) and len(x["box"]) == 4 and x.get("op") in ("add", "remove")]
        fb = [(b, op) for b, op in fb if b][:6]
        if not fb:
            break
        fp = trace(model, overlay(gray, mask, [b for b, _ in fb]), [b for b, _ in fb],
                   "focus boxes; trace the defect region to add, or the clean region to remove"); calls += 1
        for i, (b, op) in enumerate(fb):
            r = outline_mask(fp.get(i, []), b)
            mask = (mask | r) if op == "add" else (mask & ~r)
        passes += 1
    return seg, mask, {"boxes": boxes, "calls": calls, "passes": passes}


def dice(p, g):
    if not p.any() and not g.any():
        return 1.0
    return float(2 * (p & g).sum() / (p.sum() + g.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-vl:4b-instruct")
    ap.add_argument("--n-defect", type=int, default=28)
    ap.add_argument("--n-clean", type=int, default=12)
    ap.add_argument("--T", type=int, default=3)
    a = ap.parse_args()
    od = os.path.join(OUT, "armC"); os.makedirs(od, exist_ok=True)
    out = os.path.join(OUT, "armC.jsonl")
    done = {json.loads(l)["id"] for l in open(out)} if os.path.exists(out) else set()
    ids = pd.read_csv(os.path.join(SPLIT_DIR, "holdout_ids.csv"))["ImageId"].tolist()
    rng = np.random.default_rng(0); rng.shuffle(ids)
    pick, by_c, clean = [], {c: 0 for c in range(1, 5)}, 0
    for f in ids:                                   # stratified: up to n_defect/4 per class, then clean
        m = cv2.imread(os.path.join(MASK_DIR, os.path.splitext(f)[0] + ".png"), cv2.IMREAD_UNCHANGED)
        cs = [c for c in range(1, 5) if (m == c).any()]
        if cs and any(by_c[c] < a.n_defect // 4 for c in cs) and len(pick) - clean < a.n_defect:
            pick.append(f); [by_c.__setitem__(c, by_c[c] + 1) for c in cs]
        elif not cs and clean < a.n_clean:
            pick.append(f); clean += 1
    t0 = time.time()
    with open(out, "a") as fo:
        for k, f in enumerate(pick):
            if f in done:
                continue
            gray = cv2.imread(os.path.join(DATA_DIR, "train_images", f), cv2.IMREAD_GRAYSCALE)
            g = cv2.imread(os.path.join(MASK_DIR, os.path.splitext(f)[0] + ".png"), cv2.IMREAD_UNCHANGED) > 0
            t1 = time.time()
            try:
                seg, fin, info = run_one(a.model, gray, a.T)
                cv2.imwrite(os.path.join(od, os.path.splitext(f)[0] + ".png"), fin.astype(np.uint8) * 255)
                rec = {"id": f, "label_px": int(g.sum()), "seg_px": int(seg.sum()), "final_px": int(fin.sum()),
                       "dice_seg": dice(seg, g), "dice_final": dice(fin, g), **info, "sec": round(time.time() - t1, 1)}
            except Exception as e:
                rec = {"id": f, "label_px": int(g.sum()), "error": repr(e)[:300]}
            fo.write(json.dumps(rec) + "\n"); fo.flush()
            print(f"C {k + 1}/{len(pick)} {f} label {rec['label_px']} | final {rec.get('final_px')} | "
                  f"dice {rec.get('dice_final')} | calls {rec.get('calls')} | {rec.get('sec')} s", flush=True)
    print(f"C DONE in {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
