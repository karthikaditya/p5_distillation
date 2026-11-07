"""Task: intent classification over 77 banking intents (PolyAI banking77, real customer queries).
A narrow task a capable teacher model can do well -> good distillation candidate."""
import json, os, re, time
from dotenv import load_dotenv

load_dotenv()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(DATA, exist_ok=True); os.makedirs(RESULTS, exist_ok=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b")
STUDENT_BASE = os.getenv("STUDENT_BASE", "Qwen/Qwen2.5-1.5B-Instruct")

STUDENT_SYSTEM = 'Classify the banking customer message into one intent. Reply with JSON only: {"intent": "<intent_label>"}'


def load_banking77():
    from datasets import load_dataset
    last = None
    for name, kw in (("PolyAI/banking77", {"revision": "refs/convert/parquet"}), ("PolyAI/banking77", {}),
                     ("legacy-datasets/banking77", {})):
        try:
            ds = load_dataset(name, **kw)
            labels = ds["train"].features["label"].names
            # parquet mirrors sometimes drop ClassLabel names -> integers as strings would silently ruin training
            if len(labels) != 77 or all(l.isdigit() for l in labels):
                raise ValueError(f"{name}: label names missing ({labels[:3]}...)")
            return ds, labels
        except Exception as e:  # noqa
            last = e
    raise RuntimeError(f"could not load banking77: {last}")


def splits():
    """train_pool (teacher-labelled for training), heldout (gold, teacher ceiling), test (gold, untouched until Phase 4)."""
    ds, labels = load_banking77()
    train = ds["train"].shuffle(seed=42)
    heldout = train.select(range(500))
    pool = train.select(range(500, len(train)))
    return pool, heldout, ds["test"], labels


def parse_intent(text: str, labels):
    """Schema validation: must be JSON with an intent from the label set."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        intent = json.loads(m.group(0)).get("intent")
    except json.JSONDecodeError:
        return None
    return intent if intent in labels else None


def norm_text(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


# ---------------- teacher ----------------
_client = None


def teacher_classify(text: str, labels):
    """Returns (intent|None, input_tokens, output_tokens, latency_s)."""
    global _client
    from httpx import HTTPError
    from ollama import Client, ResponseError

    _client = _client or Client(host=OLLAMA_BASE_URL)
    schema = {"type": "object",
              "properties": {"intent": {"type": "string", "enum": labels}},
              "required": ["intent"], "additionalProperties": False}
    t0 = time.time()
    for attempt in range(4):
        try:
            response = _client.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": "Classify the banking customer message into exactly one of the allowed intents."},
                    {"role": "user", "content": text},
                ],
                format=schema,
                options={"temperature": 0, "num_predict": 50},
            )
            break
        except (HTTPError, ResponseError) as e:
            if attempt == 3:
                raise RuntimeError(
                    f"Ollama request failed for model {OLLAMA_MODEL!r} at {OLLAMA_BASE_URL!r}"
                ) from e
            time.sleep(2 ** attempt)
    content = response.message.content or ""
    intent = parse_intent(content, labels)
    return (intent, response.prompt_eval_count or 0, response.eval_count or 0,
            time.time() - t0)


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))] if xs else 0.0


def shingles(t: str, k: int = 3):
    w = norm_text(t).split()
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


def near_dedupe(texts, threshold: float = 0.8, num_perm: int = 64):
    """MinHash + LSH near-duplicate removal (no extra dependency). Keeps the first of each near-duplicate group."""
    import hashlib, random
    rnd = random.Random(0)
    seeds = [rnd.getrandbits(32) for _ in range(num_perm)]
    bands, rows = 16, num_perm // 16
    buckets, kept, sigs = {}, [], []
    for t in texts:
        sh = shingles(t)
        sig = [min(int.from_bytes(hashlib.blake2b(f"{s}{x}".encode(), digest_size=8).digest(), "big") for x in sh) for s in seeds]
        cands = set()
        for b in range(bands):
            key = (b, tuple(sig[b * rows:(b + 1) * rows]))
            cands |= set(buckets.get(key, []))
        dup = any(sum(a == c for a, c in zip(sig, sigs[j])) / num_perm >= threshold for j in cands)
        if dup:
            continue
        idx = len(sigs)
        sigs.append(sig)
        kept.append(t)
        for b in range(bands):
            buckets.setdefault((b, tuple(sig[b * rows:(b + 1) * rows])), []).append(idx)
    return kept
