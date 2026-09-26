"""Jev plays the same Snake games as L1 (paid, opt-in; key from the environment only, never printed or written).
  TYPESAFE_API_KEY=your-key python snake_jev.py [--probe] [--variant A|B|both]
Endpoint is a fixed constant so the key cannot be sent anywhere else. The state text is the only thing sent."""
import json, os, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from snake_env import ACTS, Snake

JEV_URL = "https://api.typesafe.ai/v1/systemone"; JEV_MODEL = "jev-latest"; USD_PER_TOKEN = 0.042 / 1e6
TOKEN_CAP = 3_000_000                                   # about $0.13; the run aborts past this
KEY = os.environ.get("TYPESAFE_API_KEY") or sys.exit("set TYPESAFE_API_KEY (environment only)")
CRIT = {"A": {"straight": "Keep going straight.", "left": "Turn left.", "right": "Turn right."},
        "B": {"straight": "Keep going straight: the way ahead is clear and the food is ahead or directly in line.",
              "left": "Turn left: the way ahead is blocked but left is clear, or the food is to the left and left is clear.",
              "right": "Turn right: the way ahead is blocked but right is clear, or the food is to the right and right is clear."}}
INSTR = {"A": "Which move should the snake make?",
         "B": "You control a snake. Choose the move that gets closer to the food without hitting a wall or the snake's own body. "
              "Never pick a move whose direction is blocked if another is clear."}
spent = {"tokens": 0, "calls": 0}

def call(state, variant):
    body = json.dumps({"state": state, "model": JEV_MODEL, "questions": {"move": {"type": "choice", "instructions": INSTR[variant], "criteria": CRIT[variant]}}}).encode()
    req = urllib.request.Request(JEV_URL, body, {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json", "User-Agent": "l1-snake-jev/1.0"})
    for attempt in range(5):
        try:
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=30) as r: data = json.load(r)
            return data, (time.perf_counter() - t0) * 1000
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504, 529) and attempt < 4: time.sleep(min(2 ** attempt, 10)); continue
            raise RuntimeError(f"Jev HTTP {e.code}") from None
        except (TimeoutError, ConnectionError, urllib.error.URLError):
            if attempt < 4: time.sleep(2 ** attempt); continue
            raise RuntimeError("Jev unreachable") from None

def find_probs(node, options):
    if isinstance(node, dict):
        hit = {k: v for k, v in node.items() if k in options and isinstance(v, (int, float))}
        if len(hit) >= 2: return hit
        for v in node.values():
            f = find_probs(v, options)
            if f: return f
    elif isinstance(node, list):
        for v in node:
            f = find_probs(v, options)
            if f: return f
    return None

def decide(state, variant):
    if spent["tokens"] > TOKEN_CAP: raise RuntimeError("token cap reached, aborting")
    data, ms = call(state, variant); spent["tokens"] += data.get("usage", {}).get("input_tokens", 0); spent["calls"] += 1
    pr = find_probs(data, list(CRIT[variant]))
    if not pr: raise RuntimeError(f"no probabilities found in the answer; top-level keys: {list(data)}")
    return ACTS.index(max(pr, key=pr.get)), ms

if "--probe" in sys.argv:
    env = Snake(10_000); a, ms = decide(env.text(), "A")
    print(f"probe ok: state '{env.text()}' -> {ACTS[a]} (expert {ACTS[env.expert()]}), {ms:.0f} ms, input tokens {spent['tokens']}, ${spent['tokens']*USD_PER_TOKEN:.6f}")
    sys.exit(0)

def play(seed, variant, max_steps=200):
    env = Snake(seed); n = agree = 0; lat = []
    while not env.done and env.steps < max_steps:
        a, ms = decide(env.text(), variant); lat.append(ms); e = env.expert(); n += 1; agree += a == e; env.step(a)
    return env.eaten, env.steps, agree / max(n, 1), lat

variants = ["A", "B"] if "--variant" not in sys.argv else (["A", "B"] if sys.argv[sys.argv.index("--variant") + 1] == "both" else [sys.argv[sys.argv.index("--variant") + 1]])
res = {}
for v in variants:
    with ThreadPoolExecutor(10) as ex: rows = list(ex.map(lambda i: play(10_000 + i, v), range(10)))
    lat = [x for r in rows for x in r[3]]
    res[v] = {"food": float(np.mean([r[0] for r in rows])), "steps": float(np.mean([r[1] for r in rows])), "agree": float(np.mean([r[2] for r in rows])),
              "median_ms": float(np.median(lat)), "p95_ms": float(np.percentile(lat, 95)), "decisions": len(lat), "tokens": spent["tokens"], "usd_so_far": spent["tokens"] * USD_PER_TOKEN}
    print(f"Jev variant {v}: food eaten {res[v]['food']:.1f}, survived {res[v]['steps']:.0f} steps, agrees with the expert on {100*res[v]['agree']:.0f}%, "
          f"latency median {res[v]['median_ms']:.0f} ms / p95 {res[v]['p95_ms']:.0f} ms, {res[v]['decisions']} decisions, spend so far ${res[v]['usd_so_far']:.4f}", flush=True)
json.dump(res, open("snake_jev_results.json", "w"))
