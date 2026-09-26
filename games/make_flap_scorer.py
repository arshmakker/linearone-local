"""Write the Flap scorer file for the L1 C binary: D = 0 (zero-shot cosine to the two option texts); all learning is TEACH lines at run time."""
import struct, sys
import numpy as np, onnxruntime as ort
from tokenizers import Tokenizer, models
A = "../snake/demo_assets"
from tokenizers import BertWordPieceTokenizer
tok = BertWordPieceTokenizer(f"{A}/vocab.txt", lowercase=True); CAP_TOK = 64 if len(sys.argv) > 1 and sys.argv[1] == "cube" else 32; tok.enable_truncation(CAP_TOK)
so = ort.SessionOptions(); so.intra_op_num_threads = 1; sess = ort.InferenceSession(f"{A}/model_int8.onnx", so, providers=["CPUExecutionProvider"])
OPT = {"noop": "Do nothing and keep falling.", "flap": "Flap upward."}
if len(sys.argv) > 1 and sys.argv[1] == "cube": OPT = {"U": "Turn the top face clockwise.", "U'": "Turn the top face anticlockwise.", "R": "Turn the right face clockwise.", "R'": "Turn the right face anticlockwise.", "F": "Turn the front face clockwise.", "F'": "Turn the front face anticlockwise."}
def emb(t):
    e = tok.encode(t); ids = np.array([e.ids], np.int64); m = np.array([e.attention_mask], np.int64)
    h = sess.run(None, {"input_ids": ids, "attention_mask": m, "token_type_ids": np.zeros_like(ids)})[0][0, 0]; return h / np.linalg.norm(h)
if __name__ == "__main__":
    C = np.stack([emb(t) for t in OPT.values()]).astype("<f4"); dim = C.shape[1]
    with open("cube.bin" if len(sys.argv) > 1 and sys.argv[1] == "cube" else "flap.bin", "wb") as f:
        f.write(b"L1S1"); f.write(struct.pack("<4I", dim, len(OPT), 1, CAP_TOK)); f.write(struct.pack("<f", 0.05))
        for n in OPT: f.write(bytes([len(n)]) + n.encode())
        f.write(np.zeros((dim, dim), "<f4").tobytes()); f.write(C.tobytes())
    print("wrote scorer")
