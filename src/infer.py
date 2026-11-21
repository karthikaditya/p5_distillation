"""Run one classification with a saved LoRA adapter.

Example:
  python -m src.infer --adapter /path/to/outputs/r8 --text "I lost my card"
"""
import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .common import STUDENT_BASE, STUDENT_SYSTEM

ap = argparse.ArgumentParser(description="Classify a message with a trained LoRA adapter.")
ap.add_argument("--adapter", required=True, help="Directory containing adapter_config.json and adapter weights")
ap.add_argument("--text", required=True, help="Customer message to classify")
ap.add_argument("--base", default=STUDENT_BASE, help="Base model used to train the adapter")
ap.add_argument("--max-new-tokens", type=int, default=24)
ap.add_argument("--no-4bit", action="store_true", help="Load the base model without 4-bit quantization")
a = ap.parse_args()

adapter_dir = Path(a.adapter).expanduser()
if not adapter_dir.is_dir() or not (adapter_dir / "adapter_config.json").is_file():
    ap.error(f"adapter directory or adapter_config.json not found: {adapter_dir}")
if not any((adapter_dir / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
    ap.error(f"adapter weights not found in: {adapter_dir}")
if a.max_new_tokens < 1:
    ap.error("--max-new-tokens must be positive")
if not torch.cuda.is_available():
    ap.error("CUDA GPU not detected. In Colab, select Runtime > Change runtime type > T4 GPU.")

compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
quantization = None
if not a.no_4bit:
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

tokenizer = AutoTokenizer.from_pretrained(a.base)
base_model = AutoModelForCausalLM.from_pretrained(
    a.base,
    torch_dtype=compute_dtype,
    device_map="auto",
    quantization_config=quantization,
)
model = PeftModel.from_pretrained(base_model, str(adapter_dir))
model.eval()

messages = [
    {"role": "system", "content": STUDENT_SYSTEM},
    {"role": "user", "content": a.text},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=a.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

print(tokenizer.decode(output[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
