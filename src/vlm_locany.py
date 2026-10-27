"""Zero-shot check, arm D: NVIDIA LocateAnything-3B (MoonViT + Qwen2.5-3B, parallel box decoding).

Runs in the isolated venv external/locany_env (transformers 4.57.1 as the model card requires). The language
model is loaded in 8-bit (bitsandbytes) so the 7.8 GB bf16 checkpoint fits the 8 GB GPU; the vision encoder,
projector and lm_head stay bf16. Hybrid mode (the default). Class-agnostic query on the first
200 holdout strips, same as arms A0 and B. No box = predicts clean.
Decoding: the model card's sampling (temperature 0.7, top-p 0.9, repetition penalty 1.1), seeded per strip.
Greedy decoding loops on repeated boxes (debug 10-27: <0><0><0><0> repeated to the token limit, even on the demo image).
"""
import json, os, re, sys, time
import numpy as np, pandas as pd, cv2, torch
from PIL import Image

ROOT = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal"
sys.path.insert(0, os.path.join(ROOT, "src"))
from train_lora import DATA_DIR, MASK_DIR, SPLIT_DIR  # noqa: E402
from transformers import AutoModel, AutoProcessor, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

MP = os.path.join(ROOT, "external", "LocateAnything-3B")
QUERY = "defect on the steel surface"
OUT = os.path.join(ROOT, "outputs", "vlm")
BOX = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")


def main(n=200, mode="hybrid"):
    out = os.path.join(OUT, "armD.jsonl")
    done = ({json.loads(l)["id"] for l in open(out) if "error" not in json.loads(l)} if os.path.exists(out) else set())
    ids = pd.read_csv(os.path.join(SPLIT_DIR, "holdout_ids.csv"))["ImageId"].tolist()[:n]
    tok = AutoTokenizer.from_pretrained(MP, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MP, trust_remote_code=True)
    q = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["vision_model", "mlp1", "lm_head"])
    model = AutoModel.from_pretrained(MP, quantization_config=q, torch_dtype=torch.bfloat16,
                                      trust_remote_code=True, device_map={"": 0}).eval()
    print(f"D loaded | GPU {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
    t0, k = time.time(), 0
    with open(out, "a") as fo:
        for i, f in enumerate(ids):
            if f in done:
                continue
            img = Image.open(os.path.join(DATA_DIR, "train_images", f)).convert("RGB")
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": f"Locate all the instances that matches the following description: {QUERY}."}]}]
            t1 = time.time()
            torch.manual_seed(i)          # sampled decoding (model-card settings), reproducible per strip
            try:
                text = proc.py_apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                ims, vids = proc.process_vision_info(msgs)
                inp = proc(text=[text], images=ims, videos=vids, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    r = model.generate(pixel_values=inp["pixel_values"].to(torch.bfloat16), input_ids=inp["input_ids"],
                                       attention_mask=inp["attention_mask"], image_grid_hws=inp.get("image_grid_hws"),
                                       tokenizer=tok, max_new_tokens=1024, use_cache=True, generation_mode=mode,
                                       do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.1)
                ans = r[0] if isinstance(r, tuple) else r
                ans = ans if isinstance(ans, str) else str(ans)
                boxes = [[int(v) for v in b] for b in BOX.findall(ans)]
                rec = {"id": f, "boxes": boxes, "answer": ans[:400], "sec": round(time.time() - t1, 2)}
            except Exception as e:
                if "out of memory" in repr(e).lower():
                    print(f"D OOM on {f} -> exit 3", flush=True); sys.exit(3)
                rec = {"id": f, "error": repr(e)[:400]}
            torch.cuda.empty_cache()
            fo.write(json.dumps(rec) + "\n"); fo.flush(); k += 1
            if k % 25 == 0 or k == 1:
                print(f"D {i + 1}/{len(ids)} | {(time.time() - t0) / k:.1f} s/strip | last {rec.get('answer', rec.get('error'))!r:.140}", flush=True)
    print(f"D DONE {k} strips in {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
