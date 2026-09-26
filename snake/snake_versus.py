"""Side by side: L1 (local C binary, taught by corrections) vs Jev (hosted API, zero-shot) playing the same Snake games live.
  python snake_versus.py                       40 s, needs TYPESAFE_API_KEY in the environment for the Jev side
  python snake_versus.py --no-jev              L1 only (no key, no network)
  python snake_versus.py --seconds 60 --teach 25 --l1-hz 30
  python snake_versus.py --no-render           no drawing, just the summary (used for tests)
Both play games from the same seeds at their real speed. The teacher is the scripted expert standing in for a human."""
import argparse, json, os, subprocess, sys, threading, time
from collections import deque
import numpy as np
from snake_env import ACTS, Snake

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=40); ap.add_argument("--teach", type=int, default=25, help="corrections L1 receives, then it plays alone")
ap.add_argument("--l1-hz", type=float, default=30, help="cap L1's moves per second so the board is watchable (0 = as fast as it can)")
ap.add_argument("--no-jev", action="store_true"); ap.add_argument("--no-render", action="store_true"); ap.add_argument("--seed", type=int, default=10_000)
ap.add_argument("--record", default="", help="write every drawn frame (time + screen text) to this JSON-lines file")
ap.add_argument("--l1-bin", default=f"{HERE}/c/l1score"); ap.add_argument("--model", default=f"{HERE}/demo_assets/model_int8.onnx")
ap.add_argument("--vocab", default=f"{HERE}/demo_assets/vocab.txt"); ap.add_argument("--scorer", default=f"{HERE}/demo_assets/snake.bin")
a = ap.parse_args()
if not a.no_jev:
    import jev_client
    if not os.environ.get("TYPESAFE_API_KEY"): sys.exit("set TYPESAFE_API_KEY for the Jev side, or pass --no-jev")

class Player:
    def __init__(s, name): s.name = name; s.food = s.games = s.best = s.moves = s.corr = 0; s.cur = 0; s.lat = []; s.stamps = deque(); s.frame = ""; s.usd = 0.0; s.err = ""; s.tokens = 0
    def note(s, ms): s.moves += 1; s.lat.append(ms); s.stamps.append(time.time())
    def rate(s):
        now = time.time()
        while s.stamps and now - s.stamps[0] > 5: s.stamps.popleft()
        return len(s.stamps) / min(5.0, max(now - T0, 0.5))
stop = threading.Event(); T0 = time.time()

def run(p, decide, teach=None, hz=0):
    seed = a.seed; env = Snake(seed)
    while not stop.is_set():
        if env.done or env.steps >= 200:
            p.games += 1; p.best = max(p.best, env.eaten); seed += 1; env = Snake(seed); p.cur = 0
        t0 = time.perf_counter()
        try: act = decide(env)
        except Exception as e: p.err = str(e); return
        p.note((time.perf_counter() - t0) * 1000)
        if teach is not None: act = teach(env, act)
        before = env.eaten; env.step(act); p.food += env.eaten - before; p.cur = env.eaten; p.frame = env.render()
        if hz: time.sleep(max(0.0, 1.0 / hz - (time.perf_counter() - t0)))

# --- L1: the real C binary over a pipe
proc = subprocess.Popen([a.l1_bin, "--stdin", a.model, a.vocab, a.scorer], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
def rpc(line): proc.stdin.write(line + "\n"); proc.stdin.flush(); return json.loads(proc.stdout.readline())
L1 = Player("L1  (local C binary, taught)")
def l1_decide(env): return ACTS.index(rpc(env.text())["label"])
def l1_teach(env, act):
    e = env.expert()
    if act != e and L1.corr < a.teach: rpc(f"TEACH\t{ACTS[e]}\t{env.text()}"); L1.corr += 1; return e
    return act
players = [(L1, l1_decide, l1_teach, a.l1_hz)]
# --- Jev: hosted API, zero-shot
JEV = Player("Jev (hosted API, zero-shot)")
if not a.no_jev:
    def jev_decide(env):
        name, tok, ms = jev_client.decide(env.text()); JEV.tokens += tok; JEV.usd = JEV.tokens * jev_client.USD_PER_TOKEN; return ACTS.index(name)
    players.append((JEV, jev_decide, None, 0))
threads = [threading.Thread(target=run, args=(p, d, t, hz), daemon=True) for p, d, t, hz in players]
T0 = time.time()
for t in threads: t.start()

def panel(p):
    lat = f"{np.median(p.lat[-200:]):.0f} ms" if p.lat else "-"
    lines = (p.frame or Snake(a.seed).render()).split("\n")
    stats = [f"food this game {p.cur:>3}   total {p.food:>4}   best {p.best:>3}", f"games {p.games:>3}   moves {p.moves:>5}",
             f"{p.rate():6.1f} decisions/s   median {lat}", f"spend ${p.usd:.4f}" + (f"   corrections taught {p.corr}" if p is L1 else "")]
    if p.err: stats.append("ERROR: " + p.err[:36])
    return [p.name.ljust(46)] + [l.ljust(46) for l in lines] + [s.ljust(46) for s in stats]
def draw():
    cols = [panel(L1)] + ([panel(JEV)] if not a.no_jev else [])
    return "\n".join("  |  ".join(c[i] for c in cols) for i in range(len(cols[0])))
rec = open(a.record, "w") if a.record else None
try:
    if not a.no_render: sys.stdout.write("\033[2J\033[?25l")
    while time.time() - T0 < a.seconds and any(t.is_alive() for t in threads):
        screen = f"L1 vs Jev on the same Snake games   {time.time()-T0:4.0f}s / {a.seconds:.0f}s\n\n" + draw()
        if not a.no_render: sys.stdout.write("\033[H" + screen + "\n"); sys.stdout.flush()
        if rec: rec.write(json.dumps({"t": time.time() - T0, "screen": screen}) + "\n")
        time.sleep(0.08)
finally:
    stop.set()
    if not a.no_render: sys.stdout.write("\033[?25h\n")
    if rec: rec.close()
    for t in threads: t.join(timeout=5)
    proc.stdin.close(); proc.wait()
dur = time.time() - T0; out = {}
print(f"\nafter {dur:.0f} s:")
for p in [L1] + ([JEV] if not a.no_jev else []):
    out[p.name] = {"food": p.food, "games": p.games, "best": p.best, "moves": p.moves, "decisions_per_s": p.moves / dur, "median_ms": float(np.median(p.lat)) if p.lat else None,
                   "usd": p.usd, "corrections": p.corr, "error": p.err}
    print(f"  {p.name:32s} food {p.food:4d}  games {p.games:3d}  best {p.best:3d}  {p.moves / dur:6.1f} decisions/s  median {out[p.name]['median_ms']:.0f} ms  spend ${p.usd:.4f}"
          + (f"  corrections {p.corr}" if p is L1 else "") + (f"  ERROR {p.err}" if p.err else ""))
json.dump(out, open(f"{HERE}/snake_versus_results.json", "w"))
