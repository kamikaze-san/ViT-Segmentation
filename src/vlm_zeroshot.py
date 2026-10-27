"""Zero-shot VLM check, arm A: local qwen3-vl:4b-instruct (Ollama) on the 500 holdout strips.

Level 0  "is there any surface defect?" -> P(yes) from the first answer token's logprobs, then a
         separate call for defect boxes as JSON in 0-1000 coordinates.
Level 1  same strip after 10 labelled example crops (2 per class + 2 clean) -> P(yes) per class,
         read at each "Type k:" answer token.
No training. Results append to outputs/vlm/armA_L{0,1}.jsonl (resume-safe); score with vlm_score.py.
"""
import argparse, base64, json, math, os, re, time, urllib.request
import numpy as np, pandas as pd, cv2

from train_lora import DATA_DIR, MASK_DIR, SPLIT_DIR

OUT = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\vlm"
URL = "http://localhost:11434/api/chat"
DESC = ("This is a grayscale image from a steel-sheet inspection camera, 256 pixels tall and 1600 pixels wide.")


def b64_file(p):
    return base64.b64encode(open(p, "rb").read()).decode()


def b64_arr(a):
    return base64.b64encode(cv2.imencode(".png", a)[1].tobytes()).decode()


THINK = None   # set from --think; None leaves the model default


def chat(model, content, images, ctx, timeout=900):
    # greedy and penalty-free so the logprobs are the model's own distribution
    body = {"model": model, "stream": False, "logprobs": True, "top_logprobs": 8,
            "options": {"num_ctx": ctx, "temperature": 0, "num_predict": 400,
                        "presence_penalty": 0, "repeat_penalty": 1.0},
            "messages": [{"role": "user", "content": content, "images": images}]}
    if THINK is not None:
        body["think"] = THINK
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def p_yes(top):
    """P(yes) among yes/no variants in one token's top_logprobs."""
    y = n = 0.0
    for t in top:
        w = t["token"].strip().lower()
        if w == "yes":
            y += math.exp(t["logprob"])
        elif w == "no":
            n += math.exp(t["logprob"])
    return y / (y + n) if y + n > 0 else float("nan")


def class_pyes(lp):
    """Walk generated tokens; after text ending in 'Type k:' the next yes/no token gives P(yes) for class k."""
    text, out = "", {}
    for tok in lp:
        m = re.search(r"Type\s*([1-4])\s*:\s*$", text)
        if m and tok["token"].strip().lower() in ("yes", "no"):
            out[int(m.group(1))] = p_yes(tok["top_logprobs"])
        text += tok["token"]
    return out


def parse_boxes(s):
    m = re.search(r"\[.*\]", s, re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    boxes = []
    for a in arr if isinstance(arr, list) else []:
        b = a.get("bbox_2d") if isinstance(a, dict) else a
        if isinstance(b, list) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
            boxes.append([float(v) for v in b])
    return boxes


def exemplars(seed=0):
    """2 crops per class (train12k images with exactly one class, cropped around its largest region) + 2 clean."""
    ids = pd.read_csv(os.path.join(SPLIT_DIR, "train12k_ids.csv"))["ImageId"].tolist()
    rng = np.random.default_rng(seed); rng.shuffle(ids)
    got, clean = {c: [] for c in range(1, 5)}, []
    for f in ids:
        m = cv2.imread(os.path.join(MASK_DIR, os.path.splitext(f)[0] + ".png"), cv2.IMREAD_UNCHANGED)
        present = [c for c in range(1, 5) if (m == c).any()]
        img = cv2.imread(os.path.join(DATA_DIR, "train_images", f), cv2.IMREAD_GRAYSCALE)
        if not present and len(clean) < 2:
            x0 = int(rng.integers(0, 1600 - 400)); clean.append(img[:, x0:x0 + 400])
        if len(present) == 1 and len(got[present[0]]) < 2:
            c = present[0]
            n, lab, st, _ = cv2.connectedComponentsWithStats((m == c).astype(np.uint8))
            if n < 2:
                continue
            k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            if st[k, cv2.CC_STAT_AREA] < 300:
                continue
            cx = st[k, cv2.CC_STAT_LEFT] + st[k, cv2.CC_STAT_WIDTH] // 2
            w = int(min(1600, max(400, st[k, cv2.CC_STAT_WIDTH] + 160)))
            x0 = int(np.clip(cx - w // 2, 0, 1600 - w))
            got[c].append(img[:, x0:x0 + w])
        if all(len(v) == 2 for v in got.values()) and len(clean) == 2:
            break
    ims, lines, k = [], [], 1
    for c in range(1, 5):
        for a in got[c]:
            ims.append(b64_arr(a)); lines.append(f"Image {k}: an example of defect Type {c} (the defect is inside this crop)."); k += 1
    for a in clean:
        ims.append(b64_arr(a)); lines.append(f"Image {k}: an example of clean steel with no defect."); k += 1
    return ims, lines, k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-vl:4b-instruct")
    ap.add_argument("--level", type=int, choices=[0, 1], required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--tag", default="armA", help="output prefix, e.g. q35 for qwen3.5:4b")
    ap.add_argument("--think", choices=["default", "off", "on"], default="default")
    a = ap.parse_args()
    global THINK
    THINK = {"default": None, "off": False, "on": True}[a.think]
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"{a.tag}_L{a.level}.jsonl")
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["id"] for l in open(out)}
    ids = pd.read_csv(os.path.join(SPLIT_DIR, "holdout_ids.csv"))["ImageId"].tolist()[:a.n]
    if a.level == 1:
        ex_imgs, ex_lines, k_strip = exemplars()
    t0, n_new = time.time(), 0
    with open(out, "a") as fo:
        for i, f in enumerate(ids):
            if f in done:
                continue
            strip = b64_file(os.path.join(DATA_DIR, "train_images", f))
            rec = {"id": f}
            try:
                if a.level == 0:
                    r = chat(a.model, DESC + " Is there any surface defect visible (for example a scratch, pit, dent, "
                             "stain, scale, crack or other damage)? Answer with exactly one word: yes or no.", [strip], 4096)
                    rec["p_yes"] = p_yes(r["logprobs"][0]["top_logprobs"]) if r.get("logprobs") else None
                    rec["answer"] = r["message"]["content"].strip()
                    r2 = chat(a.model, DESC + " Locate every surface defect. Output only a JSON list like "
                              '[{"bbox_2d": [x1, y1, x2, y2], "label": "defect"}] with coordinates on a 0-1000 scale '
                              "for both axes. Output [] if there is no defect.", [strip], 4096)
                    rec["boxes_raw"] = r2["message"]["content"][:2000]
                    rec["boxes"] = parse_boxes(r2["message"]["content"])
                else:
                    prompt = ("You will see labelled reference crops, then one strip to inspect.\n" + "\n".join(ex_lines) +
                              f"\nImage {k_strip}: the strip to inspect. " + DESC +
                              f"\nFor each defect type, does it appear anywhere in Image {k_strip}? Answer in exactly this "
                              "format and nothing else:\nType 1: yes or no\nType 2: yes or no\nType 3: yes or no\nType 4: yes or no")
                    r = chat(a.model, prompt, ex_imgs + [strip], 12288)
                    rec["answer"] = r["message"]["content"].strip()
                    rec["p_class"] = class_pyes(r.get("logprobs") or [])
                rec["prompt_tokens"] = r.get("prompt_eval_count")
            except Exception as e:
                rec["error"] = repr(e)[:300]
            fo.write(json.dumps(rec) + "\n"); fo.flush(); n_new += 1
            if n_new % 25 == 0:
                el = time.time() - t0
                print(f"L{a.level} {i + 1}/{len(ids)} | {el / n_new:.1f} s/strip | last {rec.get('answer', rec.get('error'))!r}", flush=True)
    print(f"L{a.level} DONE {n_new} new strips in {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
