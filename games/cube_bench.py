"""Cube benchmark exactly as pre-registered in PLAN.md (Game 3)."""
import json, os, subprocess, sys, warnings, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from cube_env import *
import make_flap_scorer as M
warnings.filterwarnings("ignore"); M.tok.enable_truncation(64)   # cube text is 50 tokens; the 32-token cap used for Game 1-2 would cut off three faces
HERE = os.path.dirname(os.path.abspath(__file__)); CS = f"{HERE}/../snake"; JEV = "--jev" in sys.argv; BINS = dict(a.split("=") for a in sys.argv[1:] if "=" in a)
LAT = {"coin flip": 0, "always U": 0, "decision tree": .1, "kNN (k=1)": .1, "word TF-IDF + LR": 1, "frozen bge + scaled LR": 8, "L1 nearest g100": 20, "L1 nearest g10000": 20, "Jev": 430}
def sample(seeds, depth_of): return [t for t in (scramble(s, depth_of(s)) for s in seeds) if t != SOLVED]      # a walk that returns to solved has no move to learn; dropped
EVALS = sample(range(1000, 1300), lambda s: 1 + s % 6); SOLVE = [scramble(5000 + i, 4) for i in range(20)]; HELD = set(EVALS) | set(SOLVE)
train_states = [t for t in sample(range(100_000, 102_400), lambda s: 1 + s % 6) if t not in HELD][:2000]; train_y = [optimal_moves(s)[0] for s in train_states]   # disjoint seeds AND no eval/solve state ever taught
print(f"{len(train_states)} training states, {len(EVALS)} eval states, {len(SOLVE)} solve scrambles; no state shared")
def wilson(k, n, z=1.96): p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c - h) / d, (c + h) / d
def score(pred): ok = [pred(s) in optimal_moves(s) for s in EVALS]; return sum(ok) / len(ok), wilson(sum(ok), len(ok))
def solve(pred, L):
    done = 0; moves = []
    for s0 in SOLVE:
        s = s0
        for k in range(12):
            if s == SOLVED: break
            s = apply(s, pred(s))
        if s == SOLVED: done += 1; moves.append(k if s0 != SOLVED else 0)
    return done / len(SOLVE), (np.mean(moves) if moves else float("nan"))
def memo(f):
    c = {}; return lambda s: c[s] if s in c else c.setdefault(s, f(s))
rng = np.random.default_rng(11); results = []
def report(name, pred, key, L=None):
    L = LAT[key] if L is None else L; acc, (lo, hi) = score(pred); sr, mv = solve(pred, L)
    sec = f"{mv * L / 1000:.2f}s" if not np.isnan(mv) else "-"
    print(f"{name:34s} optimal {acc:5.1%} [{lo:.1%},{hi:.1%}]  solved {sr:4.0%}  moves {mv:4.1f}  time {sec}", flush=True); results.append(dict(arm=name, optimal=acc, lo=lo, hi=hi, solved=sr, moves=None if np.isnan(mv) else mv))
report("coin flip", lambda s: MOVES[rng.integers(6)], "coin flip"); report("always U", lambda s: "U", "always U")
for N in (200, 2000):
    X = np.stack([onehot(s) for s in train_states[:N]]); y = train_y[:N]
    m = DecisionTreeClassifier(random_state=0).fit(X, y); report(f"decision tree (N={N})", memo(lambda s: m.predict([onehot(s)])[0]), "decision tree")
    m = KNeighborsClassifier(1).fit(X, y); report(f"kNN k=1 on stickers (N={N})", memo(lambda s: m.predict([onehot(s)])[0]), "kNN (k=1)")
    tx = [text(s) for s in train_states[:N]]; v = TfidfVectorizer(ngram_range=(1, 2)).fit(tx); m = LogisticRegression(C=10, max_iter=3000).fit(v.transform(tx), y); report(f"word TF-IDF + LR (N={N})", memo(lambda s: m.predict(v.transform([text(s)]))[0]), "word TF-IDF + LR")
    E_ = np.stack([M.emb(t) for t in tx]); sc = StandardScaler().fit(E_); m = LogisticRegression(C=10, max_iter=3000).fit(sc.transform(E_), y); report(f"frozen bge + scaled LR (N={N})", memo(lambda s: m.predict(sc.transform(M.emb(text(s))[None]))[0]), "frozen bge + scaled LR")
for gname, binp in (("g100", f"{CS}/c/l1score"), ("g10000", BINS.get("G10000", f"{CS}/c/l1score"))):
    for N in (25, 200):
        proc = subprocess.Popen([binp, "--stdin", f"{CS}/demo_assets/model_int8.onnx", f"{CS}/demo_assets/vocab.txt", f"{HERE}/cube.bin"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        def rpc(line, proc=proc): proc.stdin.write(line + "\n"); proc.stdin.flush(); return json.loads(proc.stdout.readline())
        for s, mv in zip(train_states[:N], train_y[:N]): rpc(f"TEACH\t{mv}\t{text(s)}")
        us = []; report(f"L1 nearest {gname} (N={N})", memo(lambda s, rpc=rpc: (lambda r: (us.append(r["us"]), r["label"])[1])(rpc(text(s)))), f"L1 nearest {gname}", L=None)
        print(f"   (L1 measured median {np.median(us) / 1000:.1f} ms per decision, 64-token cap)"); proc.stdin.close(); proc.wait()
if JEV:
    sys.path.insert(0, CS); import jev_client
    KEY = os.environ.get("TYPESAFE_API_KEY") or sys.exit("set TYPESAFE_API_KEY")
    CRIT = {mv: f"Turn the {f} face {'anticlockwise' if mv.endswith(chr(39)) else 'clockwise'}: this move brings the cube closer to solved." for mv, f in zip(MOVES, ["top", "top", "right", "right", "front", "front"])}
    CACHE = f"{HERE}/cube_jev_cache.json"; cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {"tok": 0, "ans": {}}
    def ask(t, K):
        k = f"{K}|{t}"
        if k in cache["ans"]: return cache["ans"][k]
        ex = "".join(f"State: {text(s)} -> {mv}\n" for s, mv in zip(train_states[:K], train_y[:K]))
        ins = "A 2x2x2 Rubik's cube (six faces, four stickers each). Choose the single move that brings it closer to solved." + ("\nExamples of a correct move:\n" + ex if ex else "")
        body = json.dumps({"state": t, "model": jev_client.JEV_MODEL, "questions": {"move": {"type": "choice", "instructions": ins, "criteria": CRIT}}}).encode()
        req = urllib.request.Request(jev_client.JEV_URL, body, {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json", "User-Agent": "l1-cube-jev/1.0"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=30) as r: d = json.load(r)
                pr = jev_client._find(d, list(CRIT)); cache["tok"] += d.get("usage", {}).get("input_tokens", 0); cache["ans"][k] = max(pr, key=pr.get); return cache["ans"][k]
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504, 529) and attempt < 4: time.sleep(min(2 ** attempt, 8)); continue
                raise
    for K in (0, 25):
        need = sorted({text(s) for s in EVALS + SOLVE})
        with ThreadPoolExecutor(6) as ex: list(ex.map(lambda t: ask(t, K), need))
        json.dump(cache, open(CACHE, "w"))
        report(f"Jev (K={K})", lambda s, K=K: ask(text(s), K), "Jev")
    print(f"Jev: {len(cache['ans'])} calls, {cache['tok']} input tokens = ${cache['tok'] * jev_client.USD_PER_TOKEN:.4f}")
json.dump(results, open(f"{HERE}/cube_results{'_jev' if JEV else ''}.json", "w"), indent=1)
