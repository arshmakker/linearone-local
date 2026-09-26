"""Snake: TF-IDF and frozen-BERT-style baselines vs L1, same corrections (pre-registered in RETRIES.md)."""
import json, warnings
import numpy as np, onnxruntime as ort
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
class Word:
    def __init__(s, X, y): s.v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True); s.m = LogisticRegression(C=10, max_iter=3000).fit(s.v.fit_transform(X), y)
    def predict(s, X): return s.m.predict(s.v.transform(X))
class Char:
    def __init__(s, X, y): s.v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True); s.m = LogisticRegression(C=10, max_iter=3000).fit(s.v.fit_transform(X), y)
    def predict(s, X): return s.m.predict(s.v.transform(X))
class Bert:
    def __init__(s, X, y): s.m = LogisticRegression(C=10, max_iter=3000).fit(np.stack([emb(t) for t in X]), y)
    def predict(s, X): return s.m.predict(np.stack([emb(t) for t in X]))
def play(model):
    food = steps = 0; agree = n = 0; per = {}
    for seed in SEEDS:
        env = Snake(seed)
        while not env.done and env.steps < MAX:
            t = env.text(); a = per.get(t)
            if a is None: a = per[t] = ACTS.index(model.predict([t])[0])
            n += 1; agree += a == env.expert(); env.step(a)
        food += env.eaten; steps += env.steps
    return food / len(SEEDS), steps / len(SEEDS), agree / n
y_te = np.array([s["expert"] for s in test]); X_te = [s["text"] for s in test]
print(f"44 unseen states; always-straight = {np.mean(y_te == 'straight'):.3f}\n")
print(f"{'arm':28s} K   accuracy on 44 unseen states   food eaten (5 games)   steps   agrees with expert")
for name, cls in (("word TF-IDF + LR", Word), ("char n-gram TF-IDF + LR", Char), ("frozen bge-small + LR", Bert)):
    for K in (10, 25):
        X = [c["state"] for c in corr[:K]]; y = [c["action"] for c in corr[:K]]; m = cls(X, y); acc = np.mean(m.predict(X_te) == y_te); f, s, ag = play(m)
        print(f"{name:28s} {K:2d}  {acc:26.3f}   {f:19.1f}   {s:6.0f}   {ag:.2f}")
print("\nL1 (already logged, nearest taught state):  10  0.455 / 25  0.409 accuracy; food 7.0 (K=10), 4.6 (K=25); agree 0.89 / 0.93. Scripted expert on the same seeds: 14.2 food. Opus with 25 examples on the 44 states: 1.000.")
