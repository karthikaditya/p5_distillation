"""Phase 3: LoRA / QLoRA fine-tune of a 1.5B student, everything logged to W&B.
Train two configs to get a real trade-off:
  python -m src.train --rank 8  --out outputs/r8
  python -m src.train --rank 32 --out outputs/r32 --qlora"""
import argparse, json, os
import torch, wandb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainerCallback,
                          TrainingArguments)
from .common import DATA, STUDENT_BASE, parse_intent

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, default=8)
ap.add_argument("--alpha", type=int, default=None)
ap.add_argument("--qlora", action="store_true")
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--grad-accum", type=int, default=2)
ap.add_argument("--max-len", type=int, default=256)
ap.add_argument("--val-gen", type=int, default=200, help="val examples to generate for accuracy each epoch")
ap.add_argument("--out", default="outputs/r8")
a = ap.parse_args()

run = wandb.init(project=os.getenv("WANDB_PROJECT", "distill-banking77"), name=os.path.basename(a.out), config=vars(a))
tok = AutoTokenizer.from_pretrained(STUDENT_BASE)
tok.pad_token = tok.pad_token or tok.eos_token
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
compute_dtype = torch.bfloat16 if use_bf16 else torch.float16 if torch.cuda.is_available() else torch.float32
qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype,
                          bnb_4bit_use_double_quant=True) if a.qlora else None
model = AutoModelForCausalLM.from_pretrained(STUDENT_BASE, torch_dtype=compute_dtype, device_map="auto", quantization_config=qcfg)
if a.qlora:
    model = prepare_model_for_kbit_training(model)
else:
    model.enable_input_require_grads()  # needed for gradient checkpointing with LoRA
model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha or 2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

ds = load_dataset("json", data_files={"train": os.path.join(DATA, "train.jsonl"), "val": os.path.join(DATA, "val.jsonl")})
LABELS = sorted({r["intent"] for r in ds["train"]} | {r["intent"] for r in ds["val"]})


def encode(row):
    """Loss only on the assistant answer: prompt tokens are masked with -100."""
    prompt = tok.apply_chat_template(row["messages"][:-1], tokenize=False, add_generation_prompt=True)
    full = prompt + row["messages"][-1]["content"] + tok.eos_token
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    ids = tok(full, add_special_tokens=False, truncation=True, max_length=a.max_len)["input_ids"]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [-100] * len(p_ids) + ids[len(p_ids):]}


train = ds["train"].map(encode, remove_columns=ds["train"].column_names)
val = ds["val"].map(encode, remove_columns=ds["val"].column_names)


def collate(batch):
    m = max(len(b["input_ids"]) for b in batch)
    pad = lambda xs, v: [x + [v] * (m - len(x)) for x in xs]
    return {"input_ids": torch.tensor(pad([b["input_ids"] for b in batch], tok.pad_token_id)),
            "attention_mask": torch.tensor(pad([b["attention_mask"] for b in batch], 0)),
            "labels": torch.tensor(pad([b["labels"] for b in batch], -100))}


class ValAccuracy(TrainerCallback):
    """Validation task metric per epoch + sample outputs table in W&B."""
    def on_epoch_end(self, args, state, control, model=None, **kw):
        model.eval()
        rows = ds["val"].select(range(min(a.val_gen, len(ds["val"]))))
        correct, table = 0, wandb.Table(columns=["epoch", "text", "expected", "output"])
        tok.padding_side = "left"
        for i in range(0, len(rows), 16):
            chunk = rows.select(range(i, min(i + 16, len(rows))))
            prompts = [tok.apply_chat_template(r["messages"][:-1], tokenize=False, add_generation_prompt=True) for r in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for r, t in zip(chunk, texts):
                correct += parse_intent(t, LABELS) == r["intent"]
                if len(table.data) < 20:
                    table.add_data(state.epoch, r["messages"][1]["content"], r["intent"], t)
        tok.padding_side = "right"
        acc = correct / len(rows)
        wandb.log({"val/accuracy": acc, "val/samples": table, "epoch": state.epoch})
        print(f"epoch {state.epoch:.1f} val accuracy (vs teacher labels) = {acc:.3f}")
        model.train()


args = TrainingArguments(
    output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
    per_device_train_batch_size=a.batch, per_device_eval_batch_size=a.batch, gradient_accumulation_steps=a.grad_accum,
    bf16=use_bf16, fp16=torch.cuda.is_available() and not use_bf16,
    logging_steps=10, eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
    load_best_model_at_end=True, metric_for_best_model="eval_loss", report_to=["wandb"], gradient_checkpointing=True,
    remove_unused_columns=False)
trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val, data_collator=collate, callbacks=[ValAccuracy()])
trainer.train()
model.save_pretrained(a.out)
tok.save_pretrained(a.out)
json.dump(vars(a), open(os.path.join(a.out, "train_config.json"), "w"), indent=2)
run.finish()
