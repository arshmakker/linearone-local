"""Flap benchmark exactly as pre-registered in PLAN.md. Every non-expert policy is a lookup over the 63 state texts, computed once per arm."""
import itertools, json, os, subprocess, sys, warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from flap_env import *
import make_flap_scorer as M
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); CS = f"{HERE}/../snake"; JEV = "--jev" in sys.argv
EVAL = list(range(201, 231)) if os.environ.get("TUNE") else list(range(1000, 1020)); TEACH_SEEDS = range(1, 201)
LAT = {"coin flip": 0, "always no-flap": 0, "scripted expert": 0.1, "decision tree": 0.1, "word TF-IDF + LR": 1, "frozen bge + scaled LR": 4, "L1 nearest (taught)": 12, "Jev": 430}
env0 = Flap(0)
STATES = list(itertools.product(range(9), range(7))); TEXT = {b: f"The next gap is {Flap.DY[b[0]]}. The bird is {Flap.VY[b[1]]}." for b in STATES}; assert len(set(TEXT.values())) == 63
expert = make_expert(3, 0.05)

# --- teaching: the local L1 binary flies zero-latency games, errors are corrected by the expert and remembered (PLAN: first K corrections)
proc = subprocess.Popen([os.environ.get("L1BIN", f"{CS}/c/l1score"), "--stdin", f"{CS}/demo_assets/model_int8.onnx", f"{CS}/demo_assets/vocab.txt", f"{HERE}/flap.bin"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
def rpc(line): proc.stdin.write(line + "\n"); proc.stdin.flush(); return json.loads(proc.stdout.readline())
def collect(limit):
    corr = []; games = 0
    for seed in TEACH_SEEDS:
        env = Flap(seed); games += 1
        while not env.dead and env.t < CAP and len(corr) < limit:
            t = env.text(); act = rpc(t)["label"] == "flap"; e = bool(expert(env))
            if act != e: rpc(f"TEACH\t{'flap' if e else 'noop'}\t{t}"); corr.append((t, e)); act = e
            env.step(act)
        if len(corr) >= limit: break
    return corr, games
rpc("FORGET"); CORR, TG = collect(100); print(f"teaching: {len(CORR)} corrections collected in {TG} zero-latency games (K=100 requested; there are only 63 distinct states)")
def l1_table(K):
    rpc("FORGET")
    for t, e in CORR[:K]: rpc(f"TEACH\t{'flap' if e else 'noop'}\t{t}")
    return {TEXT[b]: rpc(TEXT[b])["label"] == "flap" for b in STATES}
def fit_table(K, kind):
    tx = [c[0] for c in CORR[:K]]; y = np.array([c[1] for c in CORR[:K]])
    if len(set(y)) < 2: return {TEXT[b]: bool(y[0]) for b in STATES}
    if kind == "tfidf":
        v = TfidfVectorizer(ngram_range=(1, 2)).fit(tx); m = LogisticRegression(C=10, max_iter=3000).fit(v.transform(tx), y); return {TEXT[b]: bool(m.predict(v.transform([TEXT[b]]))[0]) for b in STATES}
    if kind == "bge":
        X = np.stack([M.emb(t) for t in tx]); sc = StandardScaler().fit(X); m = LogisticRegression(C=10, max_iter=3000).fit(sc.transform(X), y); return {TEXT[b]: bool(m.predict(sc.transform(M.emb(TEXT[b])[None]))[0]) for b in STATES}
    inv = {v: k for k, v in TEXT.items()}; X = np.array([inv[t] for t in tx]); m = DecisionTreeClassifier(random_state=0).fit(X, y); return {TEXT[b]: bool(m.predict([b])[0]) for b in STATES}
def jev_table(K):
    import jev_flap; return jev_flap.table(TEXT, CORR[:K])

def play(policy, lat, seeds=EVAL): 
    r = [run(s, policy, lat) for s in seeds]; return np.mean([x[0] for x in r]), np.mean([x[1] for x in r]), np.mean([x[1] >= CAP for x in r])
def lookup(tab): return lambda env: tab[env.text()]
rng = np.random.default_rng(7)
arms = [("coin flip", lambda env: rng.random() < 0.5, "coin flip"), ("always no-flap", lambda env: False, "always no-flap"), ("scripted expert", expert, "scripted expert")]
NK = len(CORR); Ks = [k for k in (25, 100) if k <= 100]
for K in Ks:
    kk = min(K, NK)
    for name, tab in (("decision tree", fit_table(K, "tree")), ("word TF-IDF + LR", fit_table(K, "tfidf")), ("frozen bge + scaled LR", fit_table(K, "bge")), ("L1 nearest (taught)", l1_table(K))):
        arms.append((f"{name} (K={kk})", lookup(tab), name))
    agree = lambda tab: np.mean([tab[TEXT[b]] == bool(expert_of_bucket(b)) for b in STATES]) if False else None
if JEV:
    for K in (0, 25, 100):
        arms.append((f"Jev (K={min(K, NK)})", lookup(jev_table(K)), "Jev"))
print(f"{'arm':32s} {'pipes@0ms':>9s} {'cap@0ms':>7s} | {'latency':>7s} {'pipes':>6s} {'ticks':>6s} {'cap':>5s}")
out = []
for name, pol, key in arms:
    p0, t0, c0 = play(pol, 0); L = LAT[key]; p1, t1, c1 = play(pol, L)
    print(f"{name:32s} {p0:9.1f} {c0:7.2f} | {L:5.1f}ms {p1:6.1f} {t1:6.0f} {c1:5.2f}", flush=True); out.append(dict(arm=name, pipes0=p0, cap0=c0, lat_ms=L, pipes=p1, ticks=t1, cap=c1))
json.dump(out, open(f"{HERE}/flap_results{'_jev' if JEV else ''}.json", "w"), indent=1)
