#!/usr/bin/env python3
"""CPU inference for a PEFT adapter produced by train_qwen_lora.py.

LLamaSharp serves the base GGUF only. Use this script when you want to run
the LoRA adapter without converting it back to GGUF. It never selects CUDA.
"""

from __future__ import annotations

import argparse
import json
import os


def main() -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    parser = argparse.ArgumentParser(description="CPU generate with optional Qwen LoRA adapter")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", default=None, help="PEFT adapter directory from train_qwen_lora.py")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(json.dumps({
            "device": "cpu",
            "model": args.model,
            "adapter": args.adapter,
            "dry_run": True,
        }, ensure_ascii=False))
        return 0

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter or args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.to(device)
    model.eval()

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = (
            f"<|im_start|>system\n{args.system}<|im_end|>\n"
            f"<|im_start|>user\n{args.prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[-1]:]
    print(tokenizer.decode(generated, skip_special_tokens=True).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
