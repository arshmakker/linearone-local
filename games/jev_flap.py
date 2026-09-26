"""Jev (paid, opt-in; key from the environment only) as a lookup table over the Flap state texts, with optional expert corrections in the prompt."""
import json, os, sys, urllib.request, urllib.error, time
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../snake")); import jev_client
KEY = os.environ.get("TYPESAFE_API_KEY") or sys.exit("set TYPESAFE_API_KEY")
INSTR = "You control a bird in a Flappy-style game. Gravity pulls it down; each flap lifts it. Choose whether to flap now so that it passes through the next gap."
CRIT = {"noop": "Do nothing: the bird is level with or below the gap centre line, or is already rising enough to reach it.",
        "flap": "Flap: the bird is below the gap centre line and falling, or too low to reach the gap without flapping now."}
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flap_jev_cache.json")
def ask(state, corr):
    ex = "".join(f"State: {t} -> {'flap' if e else 'noop'}\n" for t, e in corr)
    ins = INSTR + ("\nExamples of the correct action:\n" + ex if ex else "")
    body = json.dumps({"state": state, "model": jev_client.JEV_MODEL, "questions": {"act": {"type": "choice", "instructions": ins, "criteria": CRIT}}}).encode()
    req = urllib.request.Request(jev_client.JEV_URL, body, {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json", "User-Agent": "l1-flap-jev/1.0"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as r: d = json.load(r)
            pr = jev_client._find(d, list(CRIT)); return max(pr, key=pr.get) == "flap", d.get("usage", {}).get("input_tokens", 0)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504, 529) and attempt < 4: time.sleep(min(2 ** attempt, 8)); continue
            raise
def table(TEXT, corr):
    key = f"{len(corr)}"; cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    if key not in cache:
        texts = list(TEXT.values())
        with ThreadPoolExecutor(6) as ex: res = list(ex.map(lambda t: ask(t, corr), texts))
        cache[key] = {"table": {t: r[0] for t, r in zip(texts, res)}, "input_tokens": sum(r[1] for r in res)}
        json.dump(cache, open(CACHE, "w"), indent=1); print(f"Jev K={len(corr)}: {len(texts)} calls, {cache[key]['input_tokens']} input tokens = ${cache[key]['input_tokens'] * jev_client.USD_PER_TOKEN:.4f}", flush=True)
    return cache[key]["table"]
