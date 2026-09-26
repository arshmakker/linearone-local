"""Write the Snake scorer file: D = 0 (zero-shot cosine to the option texts) plus the three option embeddings; all learning
comes from TEACH lines at run time. Same int8 bge-small, one input at a time, 32-token cap."""
import struct
import numpy as np, onnxruntime as ort
from tokenizers import Tokenizer
OPT = {"straight": "Keep going straight.", "left": "Turn left.", "right": "Turn right."}
tok = Tokenizer.from_file("demo_assets/tokenizer.json"); tok.enable_truncation(32)
so = ort.SessionOptions(); so.intra_op_num_threads = 1; sess = ort.InferenceSession("demo_assets/model_int8.onnx", so, providers=["CPUExecutionProvider"])
def emb(t):
    e = tok.encode(t); ids = np.array([e.ids], np.int64); m = np.array([e.attention_mask], np.int64)
    h = sess.run(None, {"input_ids": ids, "attention_mask": m, "token_type_ids": np.zeros_like(ids)})[0][0, 0]; return h / np.linalg.norm(h)
C = np.stack([emb(t) for t in OPT.values()]).astype("<f4"); dim = C.shape[1]
with open("demo_assets/snake.bin", "wb") as f:
    f.write(b"L1S1"); f.write(struct.pack("<4I", dim, 3, 1, 32)); f.write(struct.pack("<f", 0.05))
    for n in OPT: f.write(bytes([len(n)]) + n.encode())
    f.write(np.zeros((dim, dim), "<f4").tobytes()); f.write(C.tobytes())
print("wrote demo_assets/snake.bin")
