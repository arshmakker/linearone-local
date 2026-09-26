import json, warnings
import numpy as np, onnxruntime as ort
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tokenizers import BertWordPieceTokenizer
from snake_env import ACTS, Snake
warnings.filterwarnings("ignore")
A = "demo_assets"; corr = json.load(open("corrections.json")); test = json.load(open("opus_states.json")); SEEDS = [10_000 + i for i in range(5)]; MAX = 100
tok = BertWordPieceTokenizer(f"{A}/vocab.txt", lowercase=True); tok.enable_truncation(32)
so = ort.SessionOptions(); so.intra_op_num_threads = 1; sess = ort.InferenceSession(f"{A}/model_int8.onnx", so, providers=["CPUExecutionProvider"]); cache = {}
def emb(t):
    if t not in cache:
        e = tok.encode(t); ids = np.array([e.ids], np.int64); m = np.array([e.attention_mask], np.int64)
        h = sess.run(None, {"input_ids": ids, "attention_mask": m, "token_type_ids": np.zeros_like(ids)})[0][0, 0]; cache[t] = h / np.linalg.norm(h)
    return cache[t]
y_te = np.array([s["expert"] for s in test]); X_te = np.stack([emb(s["text"]) for s in test])
for K in (10, 25):
    X = np.stack([emb(c["state"]) for c in corr[:K]]); y = [c["action"] for c in corr[:K]]; sc = StandardScaler().fit(X); m = LogisticRegression(C=10, max_iter=3000).fit(sc.transform(X), y)
    acc = np.mean(m.predict(sc.transform(X_te)) == y_te); per = {}; food = steps = ag = n = 0
    for seed in SEEDS:
        env = Snake(seed)
        while not env.done and env.steps < MAX:
            t = env.text(); a = per.get(t)
            if a is None: a = per[t] = ACTS.index(m.predict(sc.transform(emb(t)[None]))[0])
            n += 1; ag += a == env.expert(); env.step(a)
        food += env.eaten; steps += env.steps
    print(f"frozen bge-small + StandardScaler + LR  K={K}: accuracy on 44 unseen states {acc:.3f} | food {food / 5:.1f} | steps {steps / 5:.0f} | agrees with expert {ag / n:.2f}")
