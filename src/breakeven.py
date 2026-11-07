"""Report local inference operating costs; API break-even no longer applies to Ollama."""
import json, os
from .common import RESULTS

bench_path = os.path.join(RESULTS, "benchmark.json")
if not os.path.exists(bench_path):
    raise SystemExit("results/benchmark.json missing - run `python -m src.benchmark` first")

bench = json.load(open(bench_path))
student = bench.get("student")
if not student:
    raise SystemExit("benchmark.json has no student measurements")

load_path = os.path.join(RESULTS, "load_test.json")
load = json.load(open(load_path)) if os.path.exists(load_path) else None
capacity_rps = load["knee"]["throughput_rps"] if load else student["throughput_rps"]
gpu_hourly = student["gpu_hourly"]
gpu_day = gpu_hourly * 24
headroom = 0.7
capacity_per_gpu_day = capacity_rps * 86400 * headroom

print("Ollama inference has no metered LLM API charge, so API break-even is not applicable.")
print(f"Student hardware: ${gpu_hourly:.2f}/GPU-hour -> ${gpu_day:.2f}/GPU-day always-on")
print(f"Measured student hardware cost: ${student['cost_per_1k_usd']:.3f} per 1k requests")
print(f"Estimated capacity at {headroom:.0%} load: {capacity_per_gpu_day:,.0f} requests/day per GPU")

json.dump({
    "breakeven_applicable": False,
    "reason": "The teacher uses local Ollama inference and has no metered API cost.",
    "llm_api_cost_per_1k_usd": 0.0,
    "student_hardware_cost_per_1k_usd": student["cost_per_1k_usd"],
    "student_gpu_daily_cost_usd": gpu_day,
    "capacity_per_gpu_day": capacity_per_gpu_day,
}, open(os.path.join(RESULTS, "breakeven.json"), "w"), indent=2)
