"""Minimal Jev client for the Snake demos (paid, opt-in). The key comes from the environment only and is never printed or written;
the endpoint is a fixed constant so the key cannot be sent anywhere else. Only the state text is sent."""
import json, os, time, urllib.error, urllib.request

JEV_URL = "https://api.typesafe.ai/v1/systemone"; JEV_MODEL = "jev-latest"; USD_PER_TOKEN = 0.042 / 1e6
CRIT = {"straight": "Keep going straight: the way ahead is clear and the food is ahead or directly in line.",
        "left": "Turn left: the way ahead is blocked but left is clear, or the food is to the left and left is clear.",
        "right": "Turn right: the way ahead is blocked but right is clear, or the food is to the right and right is clear."}
INSTR = ("You control a snake. Choose the move that gets closer to the food without hitting a wall or the snake's own body. "
         "Never pick a move whose direction is blocked if another is clear.")

def _find(node, options):
    if isinstance(node, dict):
        hit = {k: v for k, v in node.items() if k in options and isinstance(v, (int, float))}
        if len(hit) >= 2: return hit
        for v in node.values():
            f = _find(v, options)
            if f: return f
    elif isinstance(node, list):
        for v in node:
            f = _find(v, options)
            if f: return f
    return None

def decide(state, key=None):
    """(option name, input tokens, latency ms). Raises RuntimeError on failure."""
    key = key or os.environ.get("TYPESAFE_API_KEY") or ""
    if not key: raise RuntimeError("TYPESAFE_API_KEY is not set")
    body = json.dumps({"state": state, "model": JEV_MODEL, "questions": {"move": {"type": "choice", "instructions": INSTR, "criteria": CRIT}}}).encode()
    req = urllib.request.Request(JEV_URL, body, {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "l1-snake-versus/1.0"})
    for attempt in range(5):
        try:
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=30) as r: data = json.load(r)
            pr = _find(data, list(CRIT))
            if not pr: raise RuntimeError("no probabilities in the answer")
            return max(pr, key=pr.get), data.get("usage", {}).get("input_tokens", 0), (time.perf_counter() - t0) * 1000
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504, 529) and attempt < 4: time.sleep(min(2 ** attempt, 8)); continue
            raise RuntimeError(f"Jev HTTP {e.code}") from None
        except (TimeoutError, ConnectionError, urllib.error.URLError):
            if attempt < 4: time.sleep(2 ** attempt); continue
            raise RuntimeError("Jev unreachable") from None
