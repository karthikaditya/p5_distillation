"""Phase 5: throughput vs concurrency against vLLM; find where latency starts degrading.
Run: python -m src.load_test --levels 1 2 4 8 16 32 64"""
import argparse, asyncio, json, os, time
from openai import AsyncOpenAI
from .common import RESULTS, STUDENT_SYSTEM, pct

ap = argparse.ArgumentParser()
ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
ap.add_argument("--requests", type=int, default=200)
ap.add_argument("--student", default=os.getenv("STUDENT_NAME", "student"))
a = ap.parse_args()
client = AsyncOpenAI(base_url=os.getenv("VLLM_URL", "http://localhost:8000/v1"), api_key="none")
TEXTS = ["I still have not received my new card", "Why was I charged a fee for my transfer?",
         "How do I top up with Apple Pay?", "My card payment was declined at the shop"]


async def level(c):
    sem, lats = asyncio.Semaphore(c), []
    async def one(i):
        async with sem:
            t0 = time.time()
            await client.chat.completions.create(model=a.student, temperature=0, max_tokens=24, messages=[
                {"role": "system", "content": STUDENT_SYSTEM}, {"role": "user", "content": TEXTS[i % len(TEXTS)]}])
            lats.append(time.time() - t0)
    t0 = time.time()
    await asyncio.gather(*[one(i) for i in range(a.requests)])
    return {"concurrency": c, "throughput_rps": a.requests / (time.time() - t0),
            "p50_ms": pct(lats, .5) * 1000, "p95_ms": pct(lats, .95) * 1000}


async def main():
    out = []
    for c in a.levels:
        r = await level(c)
        out.append(r)
        print(f"c={c:<3} {r['throughput_rps']:.1f} req/s  p50={r['p50_ms']:.0f}ms  p95={r['p95_ms']:.0f}ms")
    base = out[0]["p95_ms"]
    knee = next((r for r in out if r["p95_ms"] > 2 * base), out[-1])
    print(f"latency knee (p95 > 2x single-request p95) at concurrency {knee['concurrency']}")
    json.dump({"levels": out, "knee": knee}, open(os.path.join(RESULTS, "load_test.json"), "w"), indent=2)

asyncio.run(main())
