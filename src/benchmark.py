"""Phase 4: student vs teacher on the UNTOUCHED test set on three axes: quality, cost/1k, latency.
Requires vLLM running (src/serve.sh). Run: python -m src.benchmark --n 1000 --concurrency 8"""
import argparse, asyncio, json, os, time
from openai import AsyncOpenAI
from .common import splits, teacher_classify, parse_intent, pct, RESULTS, STUDENT_SYSTEM

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--concurrency", type=int, default=8)
ap.add_argument("--student", default=os.getenv("STUDENT_NAME", "student"))
ap.add_argument("--gpu-hourly", type=float, default=float(os.getenv("GPU_HOURLY_USD", "1.20")))
ap.add_argument("--utilization", type=float, default=0.35, help="realistic average GPU utilisation incl. idle time")
ap.add_argument("--skip-teacher", action="store_true")
a = ap.parse_args()

_, _, test, labels = splits()
rows = test.shuffle(seed=1).select(range(min(a.n, len(test))))
client = AsyncOpenAI(base_url=os.getenv("VLLM_URL", "http://localhost:8000/v1"), api_key="none")


async def student_one(sem, text):
    async with sem:
        t0 = time.time()
        r = await client.chat.completions.create(model=a.student, temperature=0, max_tokens=24,
                                                 messages=[{"role": "system", "content": STUDENT_SYSTEM}, {"role": "user", "content": text}])
        return parse_intent(r.choices[0].message.content, labels), time.time() - t0


async def run_student():
    sem = asyncio.Semaphore(a.concurrency)
    t0 = time.time()
    outs = await asyncio.gather(*[student_one(sem, r["text"]) for r in rows])
    return outs, time.time() - t0


async def run_teacher():
    sem = asyncio.Semaphore(a.concurrency)
    async def one(text):
        async with sem:
            return await asyncio.to_thread(teacher_classify, text, labels)
    return await asyncio.gather(*[one(r["text"]) for r in rows])


gold = [labels[r["label"]] for r in rows]
s_outs, wall = asyncio.run(run_student())
s_acc = sum(o[0] == g for o, g in zip(s_outs, gold)) / len(gold)
s_thr = len(rows) / wall  # req/s at this concurrency
s_cost_1k = a.gpu_hourly / (s_thr * 3600 * a.utilization) * 1000
res = {"n": len(rows), "concurrency": a.concurrency, "student": {
    "accuracy": s_acc, "schema_invalid": sum(o[0] is None for o in s_outs),
    "p50_ms": pct([o[1] for o in s_outs], .5) * 1000, "p95_ms": pct([o[1] for o in s_outs], .95) * 1000,
    "throughput_rps": s_thr, "cost_per_1k_usd": s_cost_1k, "gpu_hourly": a.gpu_hourly, "assumed_utilization": a.utilization}}
if not a.skip_teacher:
    t_outs = asyncio.run(run_teacher())
    t_acc = sum(o[0] == g for o, g in zip(t_outs, gold)) / len(gold)
    res["teacher"] = {"accuracy": t_acc, "p50_ms": pct([o[3] for o in t_outs], .5) * 1000,
                      "p95_ms": pct([o[3] for o in t_outs], .95) * 1000,
                      "llm_api_cost_per_1k_usd": 0.0}
    res["student"]["pct_of_teacher_quality"] = s_acc / t_acc if t_acc else None
json.dump(res, open(os.path.join(RESULTS, "benchmark.json"), "w"), indent=2)

lines = ["| | Accuracy (test) | LLM API / hardware cost per 1k req | p50 | p95 |", "|---|---|---|---|---|"]
for k in ("teacher", "student"):
    if k in res:
        r = res[k]
        cost_key = "llm_api_cost_per_1k_usd" if k == "teacher" else "cost_per_1k_usd"
        cost_basis = "API" if k == "teacher" else "GPU"
        lines.append(f"| {k} | {r['accuracy']:.1%} | ${r[cost_key]:.3f} ({cost_basis}) | {r['p50_ms']:.0f} ms | {r['p95_ms']:.0f} ms |")
open(os.path.join(RESULTS, "benchmark.md"), "w").write("\n".join(lines))
print("\n".join(lines))
print(json.dumps(res, indent=2))
