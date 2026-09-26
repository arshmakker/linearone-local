"""2x2x2 pocket cube: exact model, breadth-first distances to depth 7, and the text state. Corner DBL is fixed; moves are U, U', R, R', F, F'."""
import itertools, collections
import numpy as np
MOVES = ["U", "U'", "R", "R'", "F", "F'"]; NORMALS = {"U": (0, 1, 0), "D": (0, -1, 0), "R": (1, 0, 0), "L": (-1, 0, 0), "F": (0, 0, 1), "B": (0, 0, -1)}
COLOUR = {"U": "white", "D": "yellow", "R": "red", "L": "orange", "F": "green", "B": "blue"}
POS = sorted(itertools.product((-1, 1), repeat=3))
SLOTS = [(p, tuple(int(s * (i == a)) for i in range(3))) for p in POS for a, s in enumerate(p)]      # 24 sticker slots: (cubie position, outward normal)
IDX = {s: i for i, s in enumerate(SLOTS)}; FACE_OF = {v: k for k, v in NORMALS.items()}
def _rot(axis, sign):
    a, b = [i for i in range(3) if i != axis]; m = np.eye(3, dtype=int); m[a, a] = m[b, b] = 0; m[a, b] = -sign; m[b, a] = sign; return m
def _perm(axis, sign):
    m = _rot(axis, sign); perm = list(range(24))
    for i, (p, n) in enumerate(SLOTS):
        if p[axis] == 1: perm[IDX[(tuple(int(v) for v in m @ p), tuple(int(v) for v in m @ n))]] = i      # sticker in slot i moves to this slot
    return perm
PERMS = {mv: _perm({"U": 1, "R": 0, "F": 2}[mv[0]], -1 if mv.endswith("'") else 1) for mv in MOVES}
INV = {"U": "U'", "U'": "U", "R": "R'", "R'": "R", "F": "F'", "F'": "F"}
SOLVED = tuple(FACE_OF[n] for p, n in SLOTS)
def apply(state, mv):
    out = list(state)
    for dst, src in enumerate(PERMS[mv]): out[dst] = state[src]
    return tuple(out)
# breadth-first distances to depth 7
DIST = {SOLVED: 0}; frontier = [SOLVED]
for d in range(1, 8):
    nxt = []
    for s in frontier:
        for mv in MOVES:
            t = apply(s, mv)
            if t not in DIST: DIST[t] = d; nxt.append(t)
    frontier = nxt
def optimal_moves(s): d = DIST[s]; return [mv for mv in MOVES if DIST.get(apply(s, mv), 99) == d - 1]
def scramble(seed, depth):
    r = np.random.default_rng(seed); s = SOLVED; last = None
    for _ in range(depth):
        mv = MOVES[r.integers(6)]
        while last is not None and mv == INV[last]: mv = MOVES[r.integers(6)]
        s = apply(s, mv); last = mv
    return s
def text(s):
    lines = []
    for f in "URFDLB":
        idx = sorted((i for i, (p, n) in enumerate(SLOTS) if n == NORMALS[f]), key=lambda i: SLOTS[i][0]); lines.append(f"{f} face: " + " ".join(COLOUR[s[i]] for i in idx) + ".")
    return " ".join(lines)
def onehot(s):
    v = np.zeros(24 * 6); 
    for i, c in enumerate(s): v[i * 6 + "URFDLB".index(c)] = 1
    return v
if __name__ == "__main__":
    print(len(DIST), "states within 7 moves; per depth:", sorted(collections.Counter(DIST.values()).items()))
    for mv in MOVES: assert apply(apply(SOLVED, mv), INV[mv]) == SOLVED and apply(apply(apply(apply(SOLVED, mv), mv), mv), mv) == SOLVED
    s = scramble(1, 5); print(text(s)); print("dist", DIST[s], "optimal", optimal_moves(s))
