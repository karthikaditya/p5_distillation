# Model Distillation Pipeline

**Task:** intent classification over the 77 intents of **PolyAI banking77** (real customer messages). The task is narrow, the teacher does it well, and it has gold labels for honest measurement.
**Result:** _paste `results/benchmark.md` (quality / local hardware cost / p50 / p95 for teacher vs student), the local capacity estimate from `breakeven.py`, and the escalation rate from `router_eval.py`._

## Architecture
```mermaid
flowchart LR
  I[banking77 inputs] --> T[Ollama local teacher] --> D[Validated + deduped<br/>train / val]
  D --> L[LoRA / QLoRA<br/>Qwen2.5-7B] --> WB[W&B]
  L --> V[vLLM] --> RT{Router}
  RT -->|confident| OUT[Answer]
  RT -->|low confidence / bad JSON| T
```

## Run
```bash
cp .env.example .env && pip install -r requirements.txt
ollama list                                     # verify Ollama is running and check installed models
ollama pull qwen2.5-coder:3b                   # only if the teacher model is not installed
# On the GPU box only: pip install vllm
python -m src.teacher_eval --n 500             # Phase 1: teacher accuracy on gold held-out (ceiling)
python -m src.generate --limit 6000            # Phase 2: labels local inputs; no API cost prompt
python -m src.train --rank 8  --out outputs/r8            # Phase 3: two configs to compare
python -m src.train --rank 32 --out outputs/r32 --qlora
bash src/serve.sh outputs/r8 outputs/r32       # Phase 5: vLLM serves the trained LoRA student on :8000
python -m src.benchmark --n 1000 --concurrency 8           # Phase 4: untouched test split, 3 axes
python -m src.load_test                        # throughput vs concurrency, latency knee
python -m src.breakeven                        # local hardware cost/capacity (no API break-even)
uvicorn src.router:app --port 8010 & python -m src.router_eval --n 300   # student → teacher escalation
```

Configure `OLLAMA_MODEL` and `OLLAMA_BASE_URL` in `.env` to select the local teacher model and Ollama server. The fine-tuned student continues to be served through vLLM; Ollama's local teacher inference has no metered API charge, but still uses local hardware.

## Phases → code
| Phase | Where |
|---|---|
| 1. Narrow task + teacher ceiling | `src/common.py` (Ollama local model), `src/teacher_eval.py` |
| 2. Training set from teacher | `src/generate.py`: real inputs, schema-constrained Ollama output, invalid outputs dropped, exact **and near-duplicate** inputs removed (MinHash + LSH, Jaccard ≥ 0.8 on word 3-grams), train/val split; test split never touched. The loader refuses a dataset copy whose label names are missing |
| 3. LoRA/QLoRA student (Qwen2.5-7B-Instruct by default) | `src/train.py`: loss on answer tokens only; W&B logs loss, LR schedule, per-epoch val accuracy and a sample-output table |
| 4. 3-axis benchmark | `src/benchmark.py`: Ollama teacher API cost is $0; student hardware cost = GPU $/h ÷ (measured throughput × realistic utilisation) |
| 5. Serve + hardware estimate + router | `src/serve.sh`, `src/load_test.py`, `src/breakeven.py` (reports local hardware cost/capacity; API break-even is not applicable), `src/router.py` (escalates on schema failure or sequence confidence < threshold) |

If the student underperforms, publish that and explain why. A reported negative result is more credible than a perfect one.
