"""Escalation rate + end-to-end accuracy of the routed system on the test set.
Run (router up on :8010): python -m src.router_eval --n 300"""
import argparse, json, os
import httpx
from .common import splits, RESULTS

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--url", default="http://localhost:8010")
a = ap.parse_args()
_, _, test, labels = splits()
rows = test.shuffle(seed=2).select(range(a.n))
correct = 0
with httpx.Client(base_url=a.url, timeout=60) as c:
    for r in rows:
        correct += c.post("/classify", json={"text": r["text"]}).json()["intent"] == labels[r["label"]]
    stats = c.get("/stats").json()
res = {"n": a.n, "routed_accuracy": correct / a.n, **stats}
json.dump(res, open(os.path.join(RESULTS, "router_eval.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
