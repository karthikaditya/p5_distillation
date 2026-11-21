"""Phase 2: label real inputs with Ollama, dedupe, and split train/val.
The test split is never touched here. Run: python -m src.generate --limit 6000"""
import argparse, json, os, random, sys, time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from .common import splits, teacher_classify, norm_text, near_dedupe, DATA, RESULTS, STUDENT_SYSTEM

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=6000)
ap.add_argument("--workers", type=int, default=8)
a = ap.parse_args()

def report_progress(stage, completed, total, started):
    elapsed = time.monotonic() - started
    rate = completed / elapsed if elapsed else 0
    if total is None:
        message = f"{stage}: {completed} processed, {rate:.1f}/s"
    elif total:
        remaining = (total - completed) / rate if rate else 0
        message = (f"{stage}: {completed}/{total} ({completed / total:.0%}), "
                   f"{rate:.1f}/s, ~{remaining:.0f}s remaining")
    else:
        message = f"{stage}: 0/0"
    if sys.stderr.isatty():
        print(f"\r{message}", end="", file=sys.stderr, flush=True)
        if completed == total:
            print(file=sys.stderr, flush=True)
    else:
        print(message, file=sys.stderr, flush=True)

print("Loading dataset and preparing inputs...", flush=True)
pool, _, _, labels = splits()
# dedupe near-identical inputs before running local inference
seen, inputs = set(), []
for r in pool:
    k = norm_text(r["text"])
    if k not in seen:
        seen.add(k); inputs.append(r["text"])
exact = len(inputs)
print(f"Removing near-duplicates from {exact} inputs...", flush=True)
dedupe_started = time.monotonic()
dedupe_interval = max(1, exact // 100)
def dedupe_progress(completed, total):
    if completed % dedupe_interval == 0 or completed == total:
        report_progress("Near-duplicate removal", completed, total, dedupe_started)

inputs = near_dedupe(inputs, threshold=0.8, progress_callback=dedupe_progress)  # also drop reworded near-duplicates (Jaccard >= 0.8 on word 3-grams)
if not exact:
    report_progress("Near-duplicate removal", 0, 0, dedupe_started)
print(f"dedupe: {exact} after exact, {len(inputs)} after near-duplicate removal")
inputs = inputs[: a.limit]

print(f"Labeling {len(inputs)} unique inputs with the local Ollama model.")

label_started = time.monotonic()
label_interval = max(1, len(inputs) // 100)
outs = [None] * len(inputs)
completed = 0
with ThreadPoolExecutor(a.workers) as ex:
    futures = {ex.submit(teacher_classify, text, labels): i for i, text in enumerate(inputs)}
    print(f"Submitted {len(inputs)} requests ({a.workers} workers); "
          "waiting for Ollama responses (status every 10 seconds).", flush=True)
    pending = set(futures)
    last_report = time.monotonic()
    while pending:
        done, pending = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
        for future in done:
            outs[futures[future]] = future.result()
            completed += 1
        now = time.monotonic()
        if completed and (completed % label_interval == 0 or completed == len(inputs)):
            report_progress("Ollama labeling", completed, len(inputs), label_started)
            last_report = now
        elif not done or now - last_report >= 10:
            running = sum(future.running() for future in pending)
            queued = len(pending) - running
            elapsed = now - label_started
            status = (f"Ollama labeling: {completed}/{len(inputs)} completed; "
                      f"{running} requests running, {queued} queued; "
                      f"elapsed {elapsed:.0f}s")
            if sys.stderr.isatty():
                print(f"\r{status}", end="", file=sys.stderr, flush=True)
            else:
                print(status, file=sys.stderr, flush=True)
            last_report = now
if not inputs:
    report_progress("Ollama labeling", 0, 0, label_started)

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
