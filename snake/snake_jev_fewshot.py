"""Jev (paid, opt-in; key from the environment only) given the same first-K corrections as L1, playing the same Snake seeds."""
import json, os, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from snake_env import ACTS, Snake
import jev_client
KEY = os.environ.get("TYPESAFE_API_KEY") or sys.exit("set TYPESAFE_API_KEY"); corr = json.load(open("corrections.json")); SEEDS = [10_000 + i for i in range(5)]; MAX = 100
def decide(state, K):
    ex = "\n".join(f"State: {c['state']} -> {c['action']}" for c in corr[:K])
    body = json.dumps({"state": state, "model": jev_client.JEV_MODEL, "questions": {"move": {"type": "choice", "instructions": jev_client.INSTR + "\nExamples of the correct move:\n" + ex, "criteria": jev_client.CRIT}}}).encode()
    req = urllib.request.Request(jev_client.JEV_URL, body, {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json", "User-Agent": "l1-snake-jev-fewshot/1.0"})
    for attempt in range(5):
        try:
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=30) as r: d = json.load(r)
            pr = jev_client._find(d, list(jev_client.CRIT)); return ACTS.index(max(pr, key=pr.get)), d.get("usage", {}).get("input_tokens", 0), (time.perf_counter() - t0) * 1000
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504, 529) and attempt < 4: time.sleep(min(2 ** attempt, 8)); continue
            raise
def game(seed, K):
    env = Snake(seed); n = ag = 0; lat = []; tok = 0
    while not env.done and env.steps < MAX:
        a, t, ms = decide(env.text(), K); lat.append(ms); tok += t; n += 1; ag += a == env.expert(); env.step(a)
    return env.eaten, env.steps, ag / max(n, 1), lat, tok
for K in (10, 25):
    with ThreadPoolExecutor(5) as ex_: rows = list(ex_.map(lambda s: game(s, K), SEEDS))
    lat = [x for r in rows for x in r[3]]; tok = sum(r[4] for r in rows)
    print(f"Jev with the first {K} corrections in the prompt: food {np.mean([r[0] for r in rows]):.1f} | steps {np.mean([r[1] for r in rows]):.0f} | agrees with expert {np.mean([r[2] for r in rows]):.2f} | "
          f"median latency {np.median(lat):.0f} ms | {len(lat)} calls, {tok} input tokens = ${tok * jev_client.USD_PER_TOKEN:.4f}", flush=True)
