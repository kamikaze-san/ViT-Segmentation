"""Why does LocateAnything-3B emit degenerate <0><0><0><0> boxes? Sanity-check the 8-bit load on the model's own
demo image, then compare decoding settings (greedy vs the model card's sampling, hybrid vs slow) on one steel strip."""
import os, sys, time, torch
from PIL import Image

ROOT = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal"
MP = os.path.join(ROOT, "external", "LocateAnything-3B")
from transformers import AutoModel, AutoProcessor, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

quant = sys.argv[1] if len(sys.argv) > 1 else "8bit"
tok = AutoTokenizer.from_pretrained(MP, trust_remote_code=True)
proc = AutoProcessor.from_pretrained(MP, trust_remote_code=True)
q = (BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["vision_model", "mlp1", "lm_head"]) if quant == "8bit" else
     BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                        llm_int8_skip_modules=["vision_model", "mlp1", "lm_head"]))
model = AutoModel.from_pretrained(MP, quantization_config=q, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                  device_map={"": 0}).eval()
print(f"loaded {quant} | GPU {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)


def run(img, query, mode, sample):
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text":
            f"Locate all the instances that matches the following description: {query}."}]}]
    text = proc.py_apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ims, vids = proc.process_vision_info(msgs)
    inp = proc(text=[text], images=ims, videos=vids, return_tensors="pt").to("cuda")
    kw = dict(do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.1) if sample else dict(do_sample=False)
    t = time.time()
    with torch.no_grad():
        r = model.generate(pixel_values=inp["pixel_values"].to(torch.bfloat16), input_ids=inp["input_ids"],
                           attention_mask=inp["attention_mask"], image_grid_hws=inp.get("image_grid_hws"),
                           tokenizer=tok, max_new_tokens=384, use_cache=True, generation_mode=mode, **kw)
    a = r[0] if isinstance(r, tuple) else r
    print(f"  [{mode}, {'sample' if sample else 'greedy'}] {time.time() - t:.1f}s | {str(a)[:260]!r}", flush=True)


demo = Image.open(os.path.join(MP, "assets", "teaser.jpg")).convert("RGB")
print("demo original size", demo.size)
demo.thumbnail((1024, 1024))   # the vision sdpa path builds a dense N x N mask: 10 GiB at full size
strip = Image.open(os.path.join(ROOT, "data", "raw", "train_images", "0014fce06.jpg")).convert("RGB") \
    if os.path.exists(os.path.join(ROOT, "data", "raw", "train_images", "0014fce06.jpg")) else None
if strip is None:
    sys.path.insert(0, os.path.join(ROOT, "src")); from train_lora import DATA_DIR
    strip = Image.open(os.path.join(DATA_DIR, "train_images", "0014fce06.jpg")).convert("RGB")
print("demo image", demo.size, "| query: person")
for mode, s in (("hybrid", False), ("hybrid", True), ("slow", False)):
    run(demo, "person", mode, s)
print("steel strip 0014fce06 (labelled c3 streak) | query: defect on the steel surface")
for mode, s in (("hybrid", False), ("hybrid", True), ("slow", False), ("slow", True)):
    run(strip, "defect on the steel surface", mode, s)
