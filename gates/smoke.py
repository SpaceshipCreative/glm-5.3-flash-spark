#!/usr/bin/env python3
"""Smoke gates for a running GLM-5.3-Flash server. Stdlib only.

usage: gates/smoke.py [--url http://spark1:8000]

Checks arithmetic, a tool call, greedy determinism (and reports decode tok/s),
and that a repeated long prompt hits the prefix cache. Exit 1 on any failure.
"""
import argparse
import json
import sys
import time
import urllib.request

NO_THINK = {"enable_thinking": False}
WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def post(url, body):
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(req, timeout=3600)


def chat(base, model, content, **kw):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 512,
        "chat_template_kwargs": NO_THINK,
        **kw,
    }
    with post(f"{base}/v1/chat/completions", body) as r:
        return json.load(r)["choices"][0]["message"]


def stream(base, model, content, max_tokens):
    """Greedy streamed completion -> (text, ttft_s, decode_tok_per_s)."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": NO_THINK,
    }
    t0, first, parts, n = time.perf_counter(), None, [], 0
    with post(f"{base}/v1/chat/completions", body) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                n = ev["usage"]["completion_tokens"]
            for c in ev.get("choices", []):
                d = c.get("delta", {})
                text = d.get("content") or ""
                if first is None and (text or d.get("reasoning") or d.get("reasoning_content")):
                    first = time.perf_counter()
                parts.append(text)
    end = time.perf_counter()
    if first is None:
        raise RuntimeError("no tokens streamed")
    tps = (n - 1) / (end - first) if n > 1 and end > first else 0.0
    return "".join(parts), first - t0, tps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    base = ap.parse_args().url.rstrip("/")
    with urllib.request.urlopen(f"{base}/v1/models", timeout=30) as r:
        model = json.load(r)["data"][0]["id"]
    print(f"model: {model}")

    def arithmetic():
        out = chat(base, model, "What is 17*23? Reply with only the number.").get("content") or ""
        return "391" in out, repr(out[:80])

    def tool_call():
        m = chat(base, model, "What's the weather in Paris right now?",
                 tools=[WEATHER], tool_choice="auto")
        calls = m.get("tool_calls") or []
        if not calls:
            return False, f"no tool call, content={(m.get('content') or '')[:80]!r}"
        f = calls[0]["function"]
        args = json.loads(f["arguments"])
        return f["name"] == "get_weather" and "paris" in json.dumps(args).lower(), \
            f"{f['name']}({f['arguments']})"

    def determinism():
        prompt = "Count from 1 to 200, separated by single spaces. Output only the numbers."
        runs = [stream(base, model, prompt, 1500) for _ in range(3)]
        distinct = len({text for text, _, _ in runs})
        tps = [t for _, _, t in runs]
        ok = distinct == 1 and "199 200" in runs[0][0]
        return ok, f"{distinct} distinct of 3; decode {min(tps):.1f}-{max(tps):.1f} tok/s"

    def prefix_cache():
        # Unique header so a previous run's cache entry can't make "cold" warm.
        doc = f"[run {time.time_ns()}]\n" + "\n".join(
            f"Line {i}: the quick brown fox jumps over the lazy dog." for i in range(2000))
        prompt = doc + "\n\nHow many numbered lines are above? Reply with only the number."
        _, cold, _ = stream(base, model, prompt, 8)
        _, warm, _ = stream(base, model, prompt, 8)
        return warm < cold * 0.5, f"TTFT cold {cold:.2f}s, warm {warm:.2f}s"

    failed = 0
    for name, fn in [("arithmetic", arithmetic), ("tool_call", tool_call),
                     ("determinism", determinism), ("prefix_cache", prefix_cache)]:
        try:
            ok, info = fn()
        except Exception as e:  # report and keep going; one bad gate shouldn't hide the rest
            ok, info = False, f"{type(e).__name__}: {e}"
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name:13} {info}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
