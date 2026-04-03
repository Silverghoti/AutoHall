"""
Qwen3 local inference script following the same staged workflow as code/Llama2-Chat/main.py.

Usage example:
python code/Qwen3/main.py \
  --dataset_filepath /path/to/input.txt \
  --save_filepath /path/to/output.jsonl \
  --model_path /path/to/Qwen3-7B \
  --stage detection
"""

from typing import Optional

import fire
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(dialog):
    """Convert role/content dialog into a plain chat-style prompt string."""
    lines = []
    for msg in dialog:
        role = msg["role"].strip().capitalize()
        content = msg["content"].strip()
        lines.append(f"{role}: {content}")
    lines.append("Assistant:")
    return "\n".join(lines)


def generate_once(
    model,
    tokenizer,
    dialog,
    temperature: float,
    top_p: float,
    max_gen_len: Optional[int],
):
    prompt = build_prompt(dialog)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 0
    generated = model.generate(
        **inputs,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        max_new_tokens=max_gen_len or 256,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    output_ids = generated[0][inputs["input_ids"].shape[1]:]
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    return {
        "generation": {
            "role": "assistant",
            "content": content,
        }
    }


def main(
    dataset_filepath: str,
    save_filepath: str,
    model_path: str,
    temperature: float = 0.9,
    top_p: float = 0.9,
    max_gen_len: Optional[int] = None,
    stage: str = "detection",
    dtype: str = "bfloat16",
    trust_remote_code: bool = True,
):
    """
    staged workflow (mirrors Llama2-Chat/main.py comments):
      - hallu_data: generate one reference for each claim
      - cls: classify claim true/false with one reference
      - multi_ref: generate extra references for each claim
      - detection: compare paragraph pairs and detect conflicts (default)
    """
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

    i = 0
    p2 = None
    with open(save_filepath, "a", encoding="utf-8") as outfile:
        with open(dataset_filepath, encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.rstrip("\n")

                # step1: hallu_data
                # data = json.loads(line)
                # claim = data['claim']
                # dialogs = [[
                #     {"role": "system", "content": "Always answer one reference with complete sentence."},
                #     {"role": "user", "content": "Given one claim whose authenticity is unknown, "
                #                                "you should provide one reference about it and summarize the "
                #                                "reference in a paragraph. Claim: " + claim},
                # ]]

                # step2: cls
                # claim = line.split('""> Assistant: ')[0].split('Claim: ')[1]
                # reference = line.split('""> Assistant: ')[-1][0:-1]
                # dialogs = [[
                #     {"role": "system", "content": "Always answer true or false."},
                #     {"role": "user", "content": "Given the claim and the reference, "
                #                                "you should answer whether the claim is true or false. "
                #                                "Claim: " + claim + " Reference: " + reference},
                # ]]

                # step3: multi-reference
                # claim = line.split(' Reference: ')[0].split('Claim: ')[1]
                # dialogs = [[
                #     {"role": "system", "content": "Always answer one reference with complete sentence."},
                #     {"role": "user", "content": "Given one claim whose truthfulness is uncertain, "
                #                                "you should provide one reference about it. This reference "
                #                                "should be summarized as one paragraph. Claim: " + claim},
                # ]]

                # step4: detection
                if stage == "detection":
                    p1 = p2
                    p2 = line
                    print(p2)
                    i += 1
                    if i % 2 != 0:
                        continue

                    dialogs = [[
                        {
                            "role": "user",
                            "content": "Are there any conflicting parts in these paragraphs P1,P2? "
                                       "P1: " + p1 + " P2: " + p2,
                        }
                    ]]
                elif stage == "hallu_data":
                    data = json.loads(line)
                    claim = data["claim"]
                    dialogs = [[
                        {"role": "system", "content": "Always answer one reference with complete sentence."},
                        {
                            "role": "user",
                            "content": "Given one claim whose authenticity is unknown, "
                                       "you should provide one reference about it and summarize the reference "
                                       "in a paragraph. Claim: " + claim,
                        },
                    ]]
                elif stage == "cls":
                    claim = line.split('""> Assistant: ')[0].split('Claim: ')[1]
                    reference = line.split('""> Assistant: ')[-1][0:-1]
                    dialogs = [[
                        {"role": "system", "content": "Always answer true or false."},
                        {
                            "role": "user",
                            "content": "Given the claim and the reference, you should answer whether "
                                       "the claim is true or false. Claim: " + claim + " Reference: " + reference,
                        },
                    ]]
                elif stage == "multi_ref":
                    claim = line.split(' Reference: ')[0].split('Claim: ')[1]
                    dialogs = [[
                        {"role": "system", "content": "Always answer one reference with complete sentence."},
                        {
                            "role": "user",
                            "content": "Given one claim whose truthfulness is uncertain, you should provide "
                                       "one reference about it. This reference should be summarized as one "
                                       "paragraph. Claim: " + claim,
                        },
                    ]]
                else:
                    raise ValueError("stage must be one of: detection, hallu_data, cls, multi_ref")

                for dialog in dialogs:
                    result = generate_once(
                        model=model,
                        tokenizer=tokenizer,
                        dialog=dialog,
                        temperature=temperature,
                        top_p=top_p,
                        max_gen_len=max_gen_len,
                    )

                    for msg in dialog:
                        print(f"{msg['role'].capitalize()}: {msg['content']}\n")
                        question = "{}: {}".format(msg["role"].capitalize(), msg["content"])
                        outfile.write(json.dumps(question, ensure_ascii=False))
                    print(f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}")
                    answer = "> {}: {}".format(
                        result["generation"]["role"].capitalize(),
                        result["generation"]["content"],
                    )
                    outfile.write(json.dumps(answer, ensure_ascii=False) + "\n")
                    print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
