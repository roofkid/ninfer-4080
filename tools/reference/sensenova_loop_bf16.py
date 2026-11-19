"""One-time BF16 CPU capture for the release SenseNova-Loop-9B checkpoint.

This is a correctness reference, not a throughput path. The 19.3 GB BF16 model does not fit on
the 16 GB target GPU or in the original 15.5 GiB WSL guest. Run it only on a host with enough RAM.

Usage:
    python -m tools.reference.sensenova_loop_bf16 \
      --model /models/SenseNova-Loop-9B \
      --out profiles/reference/sensenova_loop_9b_bf16.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time
from typing import Sequence

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer


PROMPTS = (
    "Hello",
    "The capital of France is",
    "2+2=",
)


def capture(model_dir: Path, out_path: Path, max_new_tokens: int) -> dict[str, object]:
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to("cpu")
    records = []
    with torch.inference_mode():
        for prompt in PROMPTS:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            ids = tokenizer(rendered, return_tensors="pt").input_ids
            output = model.generate(
                ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
            text = tokenizer.decode(output[0, ids.shape[1] :], skip_special_tokens=False)
            logits = model(input_ids=ids, use_cache=False).logits[0, -1].float()
            records.append(
                {
                    "prompt": prompt,
                    "input_ids": ids[0].tolist(),
                    "generated_ids": output[0, ids.shape[1] :].tolist(),
                    "generated_text": text,
                    "last_logits": logits,
                    "top1": int(torch.argmax(logits)),
                }
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, out_path)
    manifest = {
        "model_dir": str(model_dir.resolve()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "threads": torch.get_num_threads(),
        "elapsed_seconds": time.perf_counter() - started,
        "prompts": [
            {
                "prompt": record["prompt"],
                "input_ids": record["input_ids"],
                "generated_ids": record["generated_ids"],
                "generated_text": record["generated_text"],
                "top1": record["top1"],
            }
            for record in records
        ],
    }
    manifest_path = Path(str(out_path) + ".json")
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("profiles/reference/sensenova_loop_9b_bf16.pt"))
    parser.add_argument("--max-new", type=int, default=8)
    args = parser.parse_args(argv)
    manifest = capture(args.model, args.out, args.max_new)
    print(json.dumps(manifest["prompts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
