"""Phase 2: label real inputs with Ollama, dedupe, and split train/val.
The test split is never touched here. Run: python -m src.generate --limit 6000"""
import argparse, json, os, random
from concurrent.futures import ThreadPoolExecutor
from .common import splits, teacher_classify, norm_text, near_dedupe, DATA, RESULTS, STUDENT_SYSTEM

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=6000)
ap.add_argument("--workers", type=int, default=8)
a = ap.parse_args()

pool, _, _, labels = splits()
# dedupe near-identical inputs before running local inference
seen, inputs = set(), []
for r in pool:
    k = norm_text(r["text"])
    if k not in seen:
        seen.add(k); inputs.append(r["text"])
exact = len(inputs)
inputs = near_dedupe(inputs, threshold=0.8)  # also drop reworded near-duplicates (Jaccard >= 0.8 on word 3-grams)
print(f"dedupe: {exact} after exact, {len(inputs)} after near-duplicate removal")
inputs = inputs[: a.limit]

print(f"Labeling {len(inputs)} unique inputs with the local Ollama model.")

with ThreadPoolExecutor(a.workers) as ex:
    outs = list(ex.map(lambda t: teacher_classify(t, labels), inputs))

kept = [{"text": t, "intent": o[0]} for t, o in zip(inputs, outs) if o[0] is not None]  # drop failures, don't teach them
random.Random(0).shuffle(kept)
n_val = max(200, len(kept) // 10)
split = {"val": kept[:n_val], "train": kept[n_val:]}
for name, rows in split.items():
    with open(os.path.join(DATA, f"{name}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps({"messages": [{"role": "system", "content": STUDENT_SYSTEM},
                                             {"role": "user", "content": r["text"]},
                                             {"role": "assistant", "content": json.dumps({"intent": r["intent"]})}],
                                "intent": r["intent"]}) + "\n")
stats = {"near_dupes_removed": exact - len(inputs) if len(inputs) < a.limit else "n/a (limit applied)", "inputs": len(inputs), "kept": len(kept), "dropped_invalid": len(inputs) - len(kept),
         "train": len(split["train"]), "val": len(split["val"]),
         "ollama_input_tokens": sum(o[1] for o in outs),
         "ollama_output_tokens": sum(o[2] for o in outs),
         "llm_api_cost_usd": 0.0}
json.dump(stats, open(os.path.join(RESULTS, "generation_stats.json"), "w"), indent=2)
print(json.dumps(stats, indent=2))
