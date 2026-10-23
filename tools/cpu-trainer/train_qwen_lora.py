#!/usr/bin/env python3
"""CPU-only LoRA SFT for Qwen2.5-0.5B-Instruct.

C# (ExternalProcessTrainingOrchestrator) launches this script. Training is not
implemented in-process in .NET. The script clears CUDA/HIP visibility before
importing torch and places the model on torch.device("cpu") in fp32.

Dataset rows are JSONL, one object per line, either:

  {"messages": [{"role": "system"|"user"|"assistant", "content": "..."}]}

or:

  {"instruction": "...", "input": "", "output": "..."}
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def force_cpu_environment() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU LoRA SFT for Qwen2.5-0.5B-Instruct")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset", required=True, help="JSONL file of messages or instruction/output rows")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, default=20, help="0 trains for --epochs instead")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the CPU plan and exit before importing torch",
    )
    return parser.parse_args()


def load_messages(path: str) -> list[list[dict[str, str]]]:
    rows: list[list[dict[str, str]]] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            messages = obj.get("messages")
            if messages is None and "instruction" in obj:
                user = obj["instruction"]
                if obj.get("input"):
                    user = f"{user}\n{obj['input']}"
                messages = [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": obj.get("output") or obj.get("response") or ""},
                ]
            if not isinstance(messages, list) or not messages:
                raise SystemExit(f"{path}:{line_number}: expected messages or instruction/output")
            rows.append(messages)
    if not rows:
        raise SystemExit(f"{path} has no examples")
    return rows


def render_chat(tokenizer, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            pass
    parts: list[str] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts) + "\n"


def main() -> int:
    force_cpu_environment()
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.dry_run:
        print(json.dumps({
            "device": "cpu",
            "model": args.model,
            "dataset": args.dataset,
            "output": args.output,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "dry_run": True,
        }, ensure_ascii=False))
        return 0

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    device = torch.device("cpu")
    if torch.cuda.is_available():
        print(
            "warning: torch.cuda.is_available() is true after hiding GPUs; tensors are still placed on CPU",
            file=sys.stderr,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.to(device)
    model.config.use_cache = False
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model.print_trainable_parameters()

    texts = [render_chat(tokenizer, messages) for messages in load_messages(args.dataset)]
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=args.max_length,
        padding="max_length",
        return_tensors="pt",
    )
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100

    class CpuDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return int(encoded["input_ids"].shape[0])

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            return {
                "input_ids": encoded["input_ids"][index],
                "attention_mask": encoded["attention_mask"][index],
                "labels": labels[index],
            }

    kwargs = {
        "output_dir": args.output,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_steps": max(args.max_steps, 1),
        "fp16": False,
        "bf16": False,
        "report_to": "none",
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
        "optim": "adamw_torch",
        "seed": 42,
        "remove_unused_columns": False,
        "gradient_checkpointing": False,
    }
    if args.max_steps > 0:
        kwargs["max_steps"] = args.max_steps

    parameters = set(TrainingArguments.__init__.__code__.co_varnames)
    if "use_cpu" in parameters:
        kwargs["use_cpu"] = True
    if "no_cuda" in parameters:
        kwargs["no_cuda"] = True

    trainer_kwargs = {
        "model": model,
        "args": TrainingArguments(**kwargs),
        "train_dataset": CpuDataset(),
    }
    trainer_params = set(Trainer.__init__.__code__.co_varnames)
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    train_result = trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    metrics = {key: float(value) if hasattr(value, "real") else value for key, value in train_result.metrics.items()}
    print("TRAINLOOP_RESULT " + json.dumps({
        "device": "cpu",
        "model": args.model,
        "output_dir": args.output,
        "metrics": metrics,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
