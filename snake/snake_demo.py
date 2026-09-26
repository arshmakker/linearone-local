"""Correction-loop demo on the real C binary: L1 plays Snake, a teacher corrects it, corrections apply on the next decision.
  python snake_demo.py            headless learning curve (measured game results at 0/10/25/50/100 corrections)
  python snake_demo.py --watch    same run, drawing the board (slowly) so it can be screen-recorded
The teacher here is the scripted expert, standing in for a human; nothing is retrained and the binary is never restarted."""
import json, subprocess, sys, time
import numpy as np
from snake_env import ACTS, Snake

WATCH = "--watch" in sys.argv; CHECK = [0, 10, 25, 50, 100]; EVAL_GAMES, MAX_STEPS = 10, 200
p = subprocess.Popen(["c/l1score", "--stdin", "bge/model_int8.onnx", "bge/vocab.txt", "c/snake.bin"],
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
def rpc(line): p.stdin.write(line + "\n"); p.stdin.flush(); return json.loads(p.stdout.readline())
lat = []
def decide(env):
    t0 = time.perf_counter(); o = rpc(env.text()); lat.append((time.perf_counter() - t0) * 1000); return ACTS.index(o["label"]), o
def play(seed, teach_budget=0, watch=False):
    env = Snake(seed); n_corr = agree = n = 0
    while not env.done and env.steps < MAX_STEPS:
        a, o = decide(env); e = env.expert(); n += 1; agree += a == e
        if a != e and teach_budget > n_corr:
            rpc(f"TEACH\t{ACTS[e]}\t{env.text()}"); n_corr += 1; a = e          # the correction is executed
        env.step(a)
        if watch:
            print("\033[2J\033[H" + env.render() + f"\n\nfood eaten {env.eaten}  step {env.steps}  corrections taught {o['examples']}  "
                  f"decision {lat[-1]:.0f} ms", flush=True); time.sleep(0.08)
    return env.eaten, env.steps, n_corr, agree / max(n, 1)
def evaluate(tag):
    r = [play(10_000 + i) for i in range(EVAL_GAMES)]; return {"food": np.mean([x[0] for x in r]), "steps": np.mean([x[1] for x in r]), "agree": np.mean([x[3] for x in r])}
def expert_games():
    out = []
    for i in range(EVAL_GAMES):
        env = Snake(10_000 + i)
        while not env.done and env.steps < MAX_STEPS: env.step(env.expert())
        out.append((env.eaten, env.steps))
    return np.mean([x[0] for x in out]), np.mean([x[1] for x in out])
def random_games():
    out = []; rng = np.random.default_rng(1)
    for i in range(EVAL_GAMES):
        env = Snake(10_000 + i)
        while not env.done and env.steps < MAX_STEPS: env.step(int(rng.integers(3)))
        out.append((env.eaten, env.steps))
    return np.mean([x[0] for x in out]), np.mean([x[1] for x in out])

rpc("FORGET"); total = 0; rows = []; seed = 0
for target in CHECK:
    while total < target:                                            # teaching games until this many corrections were given
        food, steps, c, ag = play(seed, teach_budget=target - total, watch=WATCH); seed += 1; total += c
        if c == 0 and steps >= MAX_STEPS: break
    r = evaluate(target); r["corrections"] = total; rows.append(r)
    print(f"after {total:3d} corrections: {EVAL_GAMES} fresh games, no teaching: food eaten {r['food']:.1f}, survived {r['steps']:.0f} steps, "
          f"agrees with the expert on {100*r['agree']:.0f}% of decisions", flush=True)
ef, es = expert_games(); rf, rs = random_games()
print(f"reference: scripted expert eats {ef:.1f} and survives {es:.0f} steps; random moves eat {rf:.1f} and survive {rs:.0f} steps")
print(f"decision latency incl. game logic and pipe: median {np.median(lat):.1f} ms, p95 {np.percentile(lat,95):.1f} ms over {len(lat)} decisions "
      f"(about {1000/np.median(lat):.0f} decisions per second on one thread)")
json.dump({"rows": rows, "expert": [ef, es], "random": [rf, rs], "median_ms": float(np.median(lat)), "p95_ms": float(np.percentile(lat, 95)),
           "n_decisions": len(lat)}, open("snake_demo_results.json", "w"))
p.stdin.close(); p.wait()
