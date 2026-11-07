"""Phase 5: break-even volume. Self-hosted GPUs are paid 24h/day whether busy or idle (idle time counted).
Scales to as many GPUs as the volume needs. Run: python -m src.breakeven"""
import json, math, os
from .common import RESULTS

bench_path = os.path.join(RESULTS, "benchmark.json")
if not os.path.exists(bench_path):
    raise SystemExit("results/benchmark.json missing - run `python -m src.benchmark` first")
bench = json.load(open(bench_path))
if "teacher" not in bench:
    raise SystemExit("benchmark.json has no teacher numbers (was it run with --skip-teacher?). "
                     "Re-run `python -m src.benchmark` without --skip-teacher.")
load_path = os.path.join(RESULTS, "load_test.json")
load = json.load(open(load_path)) if os.path.exists(load_path) else None

gpu_hourly = bench["student"]["gpu_hourly"]
teacher_per_req = bench["teacher"]["cost_per_1k_usd"] / 1000
capacity_rps = load["knee"]["throughput_rps"] if load else bench["student"]["throughput_rps"]
HEADROOM = 0.7  # keep 30% for peaks
cap_per_gpu_day = capacity_rps * 86400 * HEADROOM
gpu_day = gpu_hourly * 24


def self_host_cost(volume):
    gpus = max(1, math.ceil(volume / cap_per_gpu_day))
    return gpus * gpu_day, gpus


# first break-even: 1 GPU vs API
breakeven = gpu_day / teacher_per_req
# with N GPUs the step cost repeats; self-hosting stays cheaper as long as a full GPU's capacity
# is worth more on the API than the GPU costs
# worst case is just after adding GPU #2: 1 GPU of traffic must cover 2 GPUs of cost
always_cheaper_after = breakeven if cap_per_gpu_day * teacher_per_req > 2 * gpu_day else None

print(f"Teacher API: ${teacher_per_req * 1000:.3f} per 1k requests")
print(f"Student: ${gpu_hourly}/h per GPU -> ${gpu_day:.2f}/day always-on; ~{cap_per_gpu_day:,.0f} req/day per GPU at {HEADROOM:.0%} load")
if breakeven > cap_per_gpu_day:
    print(f"NOTE: break-even ({breakeven:,.0f}/day) exceeds one GPU's capacity - self-hosting never beats the API at this price/throughput.")
else:
    print(f"BREAK-EVEN: {breakeven:,.0f} requests/day (1 GPU)")
    if always_cheaper_after:
        print("Each extra GPU pays for itself once filled, so above break-even the student stays cheaper as you scale.")
print("\n| req/day | API cost/day | Self-host cost/day | GPUs | Cheaper |\n|---|---|---|---|---|")
table = []
pts = sorted({1_000, 10_000, int(breakeven), int(cap_per_gpu_day), int(cap_per_gpu_day) + 1, 100_000, 1_000_000, 10_000_000})
for v in pts:
    own, gpus = self_host_cost(v)
    api = v * teacher_per_req
    table.append({"volume": v, "api": api, "self_host": own, "gpus": gpus})
    print(f"| {v:,} | ${api:,.2f} | ${own:,.2f} | {gpus} | {'student' if own < api else 'API'} |")
json.dump({"breakeven_requests_per_day": breakeven, "capacity_per_gpu_day": cap_per_gpu_day,
           "table": table}, open(os.path.join(RESULTS, "breakeven.json"), "w"), indent=2)
