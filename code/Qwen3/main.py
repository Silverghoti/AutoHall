"""
Qwen3 local inference script for AutoHall-style experiments.

This version intentionally supports ONE raw input format only:
JSONL records shaped like:
{"claim": "...", "label": true}

Each input line is transformed into one model request and one output JSONL record.
"""

from pathlib import Path
from typing import Iterable, Optional

import fire
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ALLOWED_STAGES = {"hallu_data", "multi_ref", "cls", "detection"}


def load_claim_label_jsonl(dataset_filepath: str) -> Iterable[dict]:
    """Yield normalized records from the only supported dataset format."""
    with open(dataset_filepath, encoding="utf-8") as file:
        for line_no, raw_line in enumerate(file, start=1):
            if not raw_line.strip():
                continue
            data = json.loads(raw_line)
            if "claim" not in data or "label" not in data:
                raise ValueError(
                    f"Line {line_no} must contain keys 'claim' and 'label', got keys={list(data.keys())}"
                )
            claim = data["claim"]
            label = data["label"]
            if not isinstance(claim, str) or not claim.strip():
                raise ValueError(f"Line {line_no} has invalid claim: {claim!r}")
            if not isinstance(label, bool):
                raise ValueError(f"Line {line_no} has non-bool label: {label!r}")
            yield {
                "line_no": line_no,
                "claim": claim.strip(),
                "label": label,
            }


def build_prompt(claim: str, stage: str) -> str:
    """Build stage-specific prompt from claim only."""
    if stage == "hallu_data":
        return (
            "Given one claim whose authenticity is unknown, provide one reference about it and summarize "
            f"the reference in one paragraph. Claim: {claim}"
        )
    if stage == "multi_ref":
        return (
            "Given one claim whose truthfulness is uncertain, provide one additional reference about it. "
            f"Summarize it in one paragraph. Claim: {claim}"
        )
    if stage == "cls":
        return (
            "Judge whether the following claim is true or false. "
            "Answer with only one word: True or False. "
            f"Claim: {claim}"
        )
    # stage == "detection"
    return (
        "Analyze the claim for potential factual conflicts or internal inconsistencies. "
        "If conflicting parts are found, explain briefly; otherwise say no conflict is found. "
        f"Claim: {claim}"
    )


def generate_text(
    model,
    tokenizer,
    prompt: str,
    temperature: float,
    top_p: float,
    max_gen_len: Optional[int],
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 0
    gen_kwargs = {
        **inputs,
        "do_sample": do_sample,
        "max_new_tokens": max_gen_len or 256,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    generated = model.generate(**gen_kwargs)
    output_ids = generated[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def main(
    dataset_filepath: str,
    save_filepath: str,
    model_path: str,
    stage: str = "hallu_data",
    temperature: float = 0.9,
    top_p: float = 0.9,
    max_gen_len: Optional[int] = None,
    dtype: str = "bfloat16",
    trust_remote_code: bool = True,
):
    """
    Run local Qwen3 inference with claim+label JSONL as the only supported input format.

    Parameters:
      - dataset_filepath: input JSONL path (each line: {"claim": str, "label": bool})
      - save_filepath: output JSONL path
      - model_path: local HuggingFace model path for Qwen3
      - stage: one of hallu_data / multi_ref / cls / detection
    """
    if stage not in ALLOWED_STAGES:
        raise ValueError(f"stage must be one of {sorted(ALLOWED_STAGES)}")

    if dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    save_path = Path(save_filepath)
    if save_path.parent and not save_path.parent.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_filepath, "a", encoding="utf-8") as outfile:
        for record in load_claim_label_jsonl(dataset_filepath):
            prompt = build_prompt(record["claim"], stage)
            answer = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_gen_len=max_gen_len,
            )

            out = {
                "line_no": record["line_no"],
                "claim": record["claim"],
                "gold_label": record["label"],
                "stage": stage,
                "prompt": prompt,
                "response": answer,
            }
            outfile.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(f"[line {record['line_no']}] done")


if __name__ == "__main__":
    fire.Fire(main)
