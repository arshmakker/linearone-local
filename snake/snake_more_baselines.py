import os; os.chdir(os.path.dirname(os.path.abspath(__file__)))   # files are found next to this script
import json, re, warnings
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from snake_env import ACTS, Snake
warnings.filterwarnings("ignore")
corr = json.load(open("corrections.json")); test = json.load(open("opus_states.json")); SEEDS = [10_000 + i for i in range(5)]; MAX = 100
def feats(t):
    m = re.match(r"Food is (ahead|level|behind) and (to the left|to the right|centered)\. Blocked: ahead (yes|no), left (yes|no), right (yes|no)\.", t)
    return [["ahead", "level", "behind"].index(m[1]), ["to the left", "to the right", "centered"].index(m[2]), m[3] == "yes", m[4] == "yes", m[5] == "yes"]
def play(policy):
    food = steps = 0; agree = n = 0
    for seed in SEEDS:
        env = Snake(seed)
        while not env.done and env.steps < MAX:
            a = policy(env.text()); n += 1; agree += a == env.expert(); env.step(a)
        food += env.eaten; steps += env.steps
    return food / len(SEEDS), steps / len(SEEDS), agree / n
y_te = np.array([s["expert"] for s in test]); X_te = [s["text"] for s in test]
print(f"{'arm':44s} K   accuracy on 44 unseen states   food eaten   steps   agrees with expert")
res = []
for seed in range(200):                                                                        # coin flip = uniform random move, 200 independent random policies
    r = np.random.default_rng(seed); acc = np.mean(r.integers(0, 3, len(y_te)) == np.array([ACTS.index(v) for v in y_te])); rr = np.random.default_rng(1000 + seed)
    res.append((acc, *play(lambda t: int(rr.integers(0, 3)))))
r = np.mean(res, axis=0); lo = np.percentile([x[1] for x in res], [2.5, 97.5])
print(f"{'coin flip (uniform random over 3 moves)':44s} --  {r[0]:26.3f}   {r[1]:9.1f}   {r[2]:5.0f}   {r[3]:.2f}   (200 random policies; food 95% range {lo[0]:.1f}..{lo[1]:.1f})")
f, s, ag = play(lambda t: 0); print(f"{'always straight':44s} --  {np.mean(y_te == 'straight'):26.3f}   {f:9.1f}   {s:5.0f}   {ag:.2f}")
for K in (10, 25):
    X = np.array([feats(c["state"]) for c in corr[:K]], float); y = [c["action"] for c in corr[:K]]; m = DecisionTreeClassifier(random_state=0).fit(X, y)
    pol = lambda t, m=m: ACTS.index(m.predict(np.array([feats(t)], float))[0]); acc = np.mean(m.predict(np.array([feats(t) for t in X_te], float)) == y_te); f, s, ag = play(pol)
    print(f"{'decision tree on the 5 structured features':44s} {K:2d}  {acc:26.3f}   {f:9.1f}   {s:5.0f}   {ag:.2f}")
