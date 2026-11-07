"""Phase 5: student first, escalate to the teacher on schema failure or low confidence. Reports escalation rate.
Run: uvicorn src.router:app --port 8010"""
import math, os, time
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel
from .common import load_banking77, parse_intent, teacher_classify, STUDENT_SYSTEM

app = FastAPI(title="Distilled classifier with teacher fallback")
student = OpenAI(base_url=os.getenv("VLLM_URL", "http://localhost:8000/v1"), api_key="none")
STUDENT = os.getenv("STUDENT_NAME", "student")
THRESHOLD = float(os.getenv("ESCALATION_CONFIDENCE", "0.80"))
_, LABELS = load_banking77()
STATS = {"total": 0, "escalated": 0, "reasons": {"schema": 0, "low_confidence": 0, "student_error": 0}}


class Req(BaseModel):
    text: str


@app.post("/classify")
def classify(req: Req):
    STATS["total"] += 1
    t0 = time.time()
    reason, conf, intent = None, None, None
    try:
        r = student.chat.completions.create(model=STUDENT, temperature=0, max_tokens=24, logprobs=True,
                                            messages=[{"role": "system", "content": STUDENT_SYSTEM}, {"role": "user", "content": req.text}])
        intent = parse_intent(r.choices[0].message.content, LABELS)
        lps = [t.logprob for t in (r.choices[0].logprobs.content or [])]
        conf = math.exp(sum(lps)) if lps else 0.0  # sequence probability of the emitted JSON
        if intent is None:
            reason = "schema"
        elif conf < THRESHOLD:
            reason = "low_confidence"
    except Exception:  # noqa
        reason = "student_error"
    if reason:
        STATS["escalated"] += 1
        STATS["reasons"][reason] += 1
        intent, *_ = teacher_classify(req.text, LABELS)
    return {"intent": intent, "served_by": "teacher" if reason else "student", "escalation_reason": reason,
            "student_confidence": conf, "latency_ms": round((time.time() - t0) * 1000)}


@app.get("/stats")
def stats():
    return {**STATS, "escalation_rate": STATS["escalated"] / STATS["total"] if STATS["total"] else 0.0}
