"""Exercise the long-context retrieval and vision gates against a running ninfer-serve.

Starts no server: pass a healthy base URL. One document built from the repository's
perplexity corpus carries a single passphrase, five site codes, and an exact code block.
Each gate asks one question against that shared prefix, so only the first request pays a
full prefill and the rest report their reused token count in `timings.cache_n`. A short
code-generation request measures MTP draft acceptance at depth while the harness samples
`/metrics` and `/slots` mid-request. The final summary cross-checks the request-level
`timings` against the metrics deltas and the idle slot depth.

Usage:

    python3 -m tools.smoke.serve_retrieval \
        --base-url http://127.0.0.1:8080 --model qwen3.8-27b --output report.json
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CORPUS = _REPO_ROOT / "eval/corpora/perplexity-1m/data/pg19"
_DEFAULT_CHART = _REPO_ROOT / "examples/cli/media/visual_chart.png"

_SYSTEM = (
    "You answer questions about the document that follows. Use only the document, be "
    "exact, and do not use outside knowledge."
)
_PASSPHRASE = "QZ-7391-KESTREL"
_VAULT_CODES = [
    ("ORION", "44-A"),
    ("VEGA", "17-C"),
    ("LYRA", "92-K"),
    ("DRACO", "8-D"),
    ("PERSEUS", "61-F"),
]
_BEARING_WINDOW = "37"
_BEARING_FIELD = "phase_mdeg"
_BEARING_RETURN = "int"

_FIVE_QUESTION = (
    "The document hides five site vault codes. Reply with exactly five lines, one per "
    "site, each in the form <SITE>=<code>, for the sites ORION, VEGA, LYRA, DRACO, and "
    "PERSEUS. No other text."
)
_SINGLE_QUESTION = (
    "A secret passphrase is hidden in the document. Reply with only the passphrase, no "
    "other text."
)
_CODE_QUESTION = (
    "The document contains a C code block for a bearing telemetry driver. Reply exactly "
    "in the form BEARING_WINDOW=<value>; second_field=<name>; return_type=<type>, "
    "reporting the value assigned to BEARING_WINDOW, the name of the second struct "
    "field, and the declared return type of bearing_checksum. No other text."
)
_DECODE_PROMPT = (
    "Write a complete Python module that implements a thread-safe bounded LRU cache "
    "with per-entry TTL expiration. Include the class, a docstring, and the methods. "
    "Output only Python code, no explanation."
)


class ProbeError(RuntimeError):
    pass


def _request(
    base_url: str, method: str, path: str, payload: Any | None = None, timeout: float = 3600.0
) -> tuple[int, dict[str, Any]]:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(base_url + path, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise ProbeError(
            f"{method} {path} returned HTTP {error.code}: "
            f"{error.read().decode('utf-8', errors='replace')}"
        ) from error


def wait_for_health(base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            status, body = _request(base_url, "GET", "/health", timeout=30.0)
            if status == 200 and body == {"status": "ok"}:
                return
        except (ProbeError, urllib.error.URLError, TimeoutError) as error:
            last_error = error
        time.sleep(0.25)
    raise ProbeError(f"server did not become healthy within {timeout:g}s: {last_error}")


def count_tokens(base_url: str, model: str, system_text: str, user_text: str) -> int:
    _, body = _request(
        base_url,
        "POST",
        "/v1/messages/count_tokens",
        {"model": model, "system": system_text, "messages": [{"role": "user", "content": user_text}]},
        timeout=120.0,
    )
    tokens = body.get("input_tokens")
    if not isinstance(tokens, int) or tokens <= 0:
        raise ProbeError(f"count_tokens returned {body!r}")
    return tokens


def _plant(text: str, insertions: list[tuple[float, str]]) -> str:
    pieces: list[str] = []
    previous = 0
    for fraction, snippet in sorted(insertions, key=lambda item: item[0]):
        index = min(len(text), int(len(text) * fraction))
        newline = text.find("\n", index)
        if newline != -1:
            index = newline
        pieces.append(text[previous:index])
        pieces.append(f"\n\n{snippet}\n\n")
        previous = index
    pieces.append(text[previous:])
    return "".join(pieces)

def build_document(corpus_dir: Path, target_tokens: int, base_url: str, model: str) -> str:
    text = ""
    for path in sorted(corpus_dir.glob("*.txt")):
        text += path.read_text(encoding="utf-8") + "\n\n"
    if not text:
        raise ProbeError(f"no corpus text under {corpus_dir}")

    code_block = (
        "/* bearing telemetry driver */\n"
        f"#define BEARING_WINDOW {_BEARING_WINDOW}\n"
        "typedef struct BearingSample {\n"
        "    uint16_t rpm;\n"
        f"    int16_t  {_BEARING_FIELD};\n"
        "} BearingSample;\n"
        "int bearing_checksum(const BearingSample* sample, size_t count);"
    )
    insertions = [(0.55, f"The secret passphrase is {_PASSPHRASE}.")]
    insertions += [
        (0.10 + 0.18 * index, f"The vault code for site {site} is {code}.")
        for index, (site, code) in enumerate(_VAULT_CODES)
    ]
    insertions.append((0.74, code_block))

    def planted(base_chars: int) -> str:
        return _plant(text[:base_chars], insertions)

    def measure(base_chars: int) -> int:
        return count_tokens(base_url, model, planted(base_chars), _FIVE_QUESTION)

    low, high = 0, len(text)
    best = 0
    for _ in range(8):
        middle = (low + high) // 2
        if measure(middle) <= target_tokens:
            best = middle
            low = middle
        else:
            high = middle
    document = planted(best)
    actual = measure(best)
    print(
        f"document: {len(document)} chars, {actual} prompt tokens "
        f"(target {target_tokens})",
        file=sys.stderr,
    )
    return document


def chat(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    *,
    enable_thinking: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    _, body = _request(
        base_url,
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": 0,
            "enable_thinking": enable_thinking,
        },
    )
    wall = time.monotonic() - started
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ProbeError(f"chat response has no single choice: {body!r}")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ProbeError("chat response has no message")
    return {
        "content": message.get("content") or "",
        "reasoning": message.get("reasoning_content") or "",
        "timings": body.get("timings") or {},
        "usage": body.get("usage") or {},
        "finish_reason": choices[0].get("finish_reason"),
        "id_slot": body.get("id_slot"),
        "session_digest": body.get("session_digest"),
        "wall_seconds": wall,
    }


def get_text(base_url: str, path: str) -> str:
    request = urllib.request.Request(base_url + path, headers={"Accept": "text/plain"})
    with urllib.request.urlopen(request, timeout=120.0) as response:
        return response.read().decode("utf-8")


def parse_metrics(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) == 2:
            try:
                values[parts[0]] = float(parts[1])
            except ValueError:
                continue
    return values


def _five_lines(content: str) -> dict[str, str]:
    codes: dict[str, str] = {}
    for site in (site for site, _ in _VAULT_CODES):
        match = re.search(rf"{site}\s*=\s*([A-Za-z0-9\-]+)", content, re.IGNORECASE)
        if match:
            codes[site] = match.group(1)
    return codes


def _fields(content: str, keys: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for key in keys:
        match = re.search(rf"{key}\s*[=:]\s*([^\s;,]+)", content, re.IGNORECASE)
        if match:
            values[key] = match.group(1)
    return values


def run(base_url: str, model: str, corpus_dir: Path, chart: Path, target_tokens: int) -> dict[str, Any]:
    wait_for_health(base_url, 300.0)
    document = build_document(corpus_dir, target_tokens, base_url, model)
    system = {"role": "system", "content": f"{_SYSTEM}\n\n{document}"}
    probes: list[dict[str, Any]] = []

    def user(question: str) -> list[dict[str, Any]]:
        return [system, {"role": "user", "content": question}]

    metrics_before = parse_metrics(get_text(base_url, "/metrics"))

    def record(name: str, result: dict[str, Any], passed: bool, detail: Any) -> None:
        timings = result["timings"]
        probes.append(
            {
                "name": name,
                "pass": passed,
                "answer": result["content"].strip(),
                "detail": detail,
                "prompt_tokens": result["usage"].get("prompt_tokens"),
                "cache_n": timings.get("cache_n"),
                "prompt_n": timings.get("prompt_n"),
                "predicted_n": timings.get("predicted_n"),
                "predicted_per_second": timings.get("predicted_per_second"),
                "draft_n": timings.get("draft_n"),
                "draft_n_accepted": timings.get("draft_n_accepted"),
                "wall_seconds": result["wall_seconds"],
                "id_slot": result["id_slot"],
            }
        )
        print(
            f"[{name}] {'PASS' if passed else 'FAIL'} "
            f"prompt={timings.get('prompt_n')} cache={timings.get('cache_n')} "
            f"predicted={timings.get('predicted_n')} "
            f"decode={timings.get('predicted_per_second'):.1f} tok/s "
            f"wall={result['wall_seconds']:.1f}s\n  {result['content'].strip()[:240]}",
            file=sys.stderr,
        )

    single = chat(base_url, model, user(_SINGLE_QUESTION), 32)
    record("single_needle", single, _PASSPHRASE in single["content"], _PASSPHRASE)

    five = chat(base_url, model, user(_FIVE_QUESTION), 96)
    found = _five_lines(five["content"])
    expected = dict(_VAULT_CODES)
    five_ok = all(found.get(site, "").upper() == code.upper() for site, code in expected.items())
    record("five_needles", five, five_ok, {"found": found, "expected": expected})

    code = chat(base_url, model, user(_CODE_QUESTION), 48)
    fields = _fields(code["content"], ["BEARING_WINDOW", "second_field", "return_type"])
    code_ok = (
        fields.get("BEARING_WINDOW") == _BEARING_WINDOW
        and fields.get("second_field", "").lower() == _BEARING_FIELD
        and fields.get("return_type", "").lower() == _BEARING_RETURN
    )
    record("code_detail", code, code_ok, fields)

    # A decode-heavy request at depth, sampled for /metrics and /slots truthfulness.
    busy: dict[str, Any] = {}
    decode_result: dict[str, Any] = {}
    decode_error: list[BaseException] = []

    def decode_request() -> None:
        try:
            decode_result.update(
                chat(base_url, model, user(_DECODE_PROMPT), 256, enable_thinking=False)
            )
        except BaseException as error:  # surfaced on the main thread
            decode_error.append(error)

    worker = threading.Thread(target=decode_request)
    worker.start()
    deadline = time.monotonic() + 600.0
    while worker.is_alive() and time.monotonic() < deadline:
        metrics = parse_metrics(get_text(base_url, "/metrics"))
        if metrics.get("llamacpp:requests_processing", 0) >= 1:
            slots = json.loads(get_text(base_url, "/slots"))
            busy = {"metrics": metrics, "slots": slots}
            if any(slot.get("is_processing") and slot.get("n_prompt_tokens_cache") for slot in slots):
                break
        time.sleep(0.2)
    worker.join()
    if decode_error:
        raise decode_error[0]
    timings = decode_result["timings"]
    draft_n = timings.get("draft_n") or 0
    accepted = timings.get("draft_n_accepted") or 0
    acceptance = (accepted / draft_n) if draft_n else None
    record(
        "decode_at_depth",
        decode_result,
        bool(decode_result["content"]) and bool(decode_result["finish_reason"] == "length"),
        {"acceptance": acceptance},
    )

    time.sleep(0.5)
    slots_idle = json.loads(get_text(base_url, "/slots"))
    metrics_after = parse_metrics(get_text(base_url, "/metrics"))

    image = base64.b64encode(chart.read_bytes()).decode("ascii")
    vision = chat(
        base_url,
        model,
        [
            {"role": "system", "content": "You answer questions about the image that follows."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Read the chart. Reply exactly in the form "
                            "NUMBER=<title number>; CIRCLES=<red circle count>; "
                            "SIDE=<left|right: which side of the green triangle the blue "
                            "square is on>. No other text."
                        ),
                    },
                ],
            },
        ],
        48,
    )
    vision_fields = _fields(vision["content"], ["NUMBER", "CIRCLES", "SIDE"])
    vision_ok = (
        vision_fields.get("NUMBER") == "731"
        and vision_fields.get("CIRCLES") == "3"
        and vision_fields.get("SIDE", "").lower() == "left"
    )
    record("vision_chart", vision, vision_ok, vision_fields)

    metrics_final = parse_metrics(get_text(base_url, "/metrics"))
    slots_final = json.loads(get_text(base_url, "/slots"))

    deltas = {
        key: round(value - metrics_before.get(key, 0.0), 6)
        for key, value in metrics_final.items()
        if value != metrics_before.get(key)
    }
    expected_prompt = sum(
        probe["prompt_n"] or 0 for probe in probes
    )
    expected_cache = sum(probe["cache_n"] or 0 for probe in probes)
    expected_draft = sum((probe["draft_n"] or 0) for probe in probes)
    expected_accepted = sum((probe["draft_n_accepted"] or 0) for probe in probes)
    metrics_ok = (
        abs(deltas.get("llamacpp:prompt_tokens_total", 0) - expected_prompt) < 0.5
        and abs(deltas.get("ninfer:prefix_cache_hit_tokens_total", 0) - expected_cache) < 0.5
        and abs(deltas.get("ninfer:requests_total", 0) - len(probes)) < 0.5
        and abs(deltas.get("ninfer:draft_tokens_total", 0) - expected_draft) < 0.5
        and abs(deltas.get("ninfer:draft_accepted_tokens_total", 0) - expected_accepted) < 0.5
    )
    retained = [
        {
            "id": slot["id"],
            "retained": slot["retained"],
            "n_prompt_tokens": slot["n_prompt_tokens"],
            "session_digest": slot["session_digest"],
        }
        for slot in slots_final
        if slot.get("retained")
    ]
    rows = slots_idle[0] if slots_idle else {}

    summary = {
        "format": "ninfer_serve_retrieval_v1",
        "model": model,
        "document_chars": len(document),
        "probes": probes,
        "decode": {
            "draft_n": draft_n,
            "draft_n_accepted": accepted,
            "acceptance": acceptance,
            "predicted_per_second": timings.get("predicted_per_second"),
        },
        "busy_sample": busy,
        "slots_idle": rows,
        "retained_slots": retained,
        "metrics_deltas": deltas,
        "metrics_expected": {
            "llamacpp:prompt_tokens_total": expected_prompt,
            "ninfer:prefix_cache_hit_tokens_total": expected_cache,
            "ninfer:requests_total": len(probes),
            "ninfer:draft_tokens_total": expected_draft,
            "ninfer:draft_accepted_tokens_total": expected_accepted,
        },
        "metrics_ok": metrics_ok,
        "all_pass": all(probe["pass"] for probe in probes) and metrics_ok,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", required=True)
    parser.add_argument("--corpus", type=Path, default=_DEFAULT_CORPUS)
    parser.add_argument("--chart", type=Path, default=_DEFAULT_CHART)
    parser.add_argument("--target-tokens", type=int, default=99000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = run(
        args.base_url.rstrip("/"),
        args.model,
        args.corpus,
        args.chart,
        args.target_tokens,
    )
    if args.output:
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["all_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
