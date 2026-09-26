"""10x10 Snake with a scripted safe-greedy expert; states are short text (same phrasing as learnability.py)."""
import numpy as np
W = H = 10; ACTS = ["straight", "left", "right"]
def turn(d, a): return d if a == 0 else ((d[1], -d[0]) if a == 1 else (-d[1], d[0]))

class Snake:
    def __init__(self, seed): self.rng = np.random.default_rng(seed); self.reset()
    def reset(self):
        self.body = [(5, 5), (4, 5), (3, 5)]; self.d = (1, 0); self.eaten = 0; self.steps = 0; self.hungry = 0; self.done = False; self._food()
    def _food(self):
        while True:
            self.food = (int(self.rng.integers(W)), int(self.rng.integers(H)))
            if self.food not in self.body: return
    def bad(self, p): return not (0 <= p[0] < W and 0 <= p[1] < H) or p in self.body[:-1]
    def nxt(self, a): nd = turn(self.d, a); return (self.body[0][0] + nd[0], self.body[0][1] + nd[1])
    def text(self):
        hx, hy = self.body[0]; rx, ry = self.food[0] - hx, self.food[1] - hy; f = rx * self.d[0] + ry * self.d[1]
        left = (self.d[1], -self.d[0]); lat = rx * left[0] + ry * left[1]
        fw = "ahead" if f > 0 else "behind" if f < 0 else "level"; lt = "to the left" if lat > 0 else "to the right" if lat < 0 else "centered"
        bl = ["yes" if self.bad(self.nxt(a)) else "no" for a in range(3)]
        return f"Food is {fw} and {lt}. Blocked: ahead {bl[0]}, left {bl[1]}, right {bl[2]}."
    def expert(self):
        best, ba = None, 0
        for a in range(3):
            p = self.nxt(a)
            if self.bad(p): continue
            dist = abs(p[0] - self.food[0]) + abs(p[1] - self.food[1])
            if best is None or dist < best: best, ba = dist, a
        return ba
    def step(self, a):
        p = self.nxt(a); self.steps += 1; self.hungry += 1
        if self.bad(p) or self.hungry > 100: self.done = True; return
        self.d = turn(self.d, a); self.body = [p] + self.body
        if p == self.food: self.eaten += 1; self.hungry = 0; self._food()
        else: self.body = self.body[:-1]
    def render(self):
        g = [["." for _ in range(W)] for _ in range(H)]
        for i, (x, y) in enumerate(self.body): g[y][x] = "@" if i == 0 else "o"
        g[self.food[1]][self.food[0]] = "*"
        return "\n".join(" ".join(r) for r in g)
