"""Phase 1: teacher accuracy on a gold held-out set - the ceiling for the student.
Run: python -m src.teacher_eval [--n 500]"""
import argparse, json, os
from concurrent.futures import ThreadPoolExecutor
from .common import splits, teacher_classify, pct, RESULTS, OLLAMA_MODEL

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--workers", type=int, default=8)
a = ap.parse_args()
_, heldout, _, labels = splits()
rows = heldout.select(range(min(a.n, len(heldout))))
with ThreadPoolExecutor(a.workers) as ex:
    outs = list(ex.map(lambda r: teacher_classify(r["text"], labels), rows))
correct = sum(o[0] == labels[r["label"]] for o, r in zip(outs, rows))
in_tok, out_tok = sum(o[1] for o in outs), sum(o[2] for o in outs)
res = {"teacher": OLLAMA_MODEL, "n": len(rows), "accuracy": correct / len(rows),
       "invalid": sum(o[0] is None for o in outs), "input_tokens": in_tok, "output_tokens": out_tok,
       "llm_api_cost_usd": 0.0,
       "p50_s": pct([o[3] for o in outs], .5), "p95_s": pct([o[3] for o in outs], .95)}
json.dump(res, open(os.path.join(RESULTS, "teacher_eval.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
