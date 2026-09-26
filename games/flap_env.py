"""Flap: a deterministic Flappy-style game with a real-time clock. Constants are fixed in PLAN.md before any learner is evaluated."""
import numpy as np
TICK_MS = 1000 / 30; G = 0.0016; FLAP_V = -0.021; SPEED = 0.008; SPACING = 0.72; HALF_GAP = 0.17; PIPE_W = 0.08; BIRD_X = 0.25; BIRD_R = 0.02; CAP = 900

class Flap:
    def __init__(self, seed):
        r = np.random.default_rng(seed); self.gaps = r.uniform(0.30, 0.70, 40)          # gap centres, one per pipe
        self.y, self.vy, self.t, self.passed, self.dead = 0.5, 0.0, 0, 0, False
    def pipe_x(self, k): return 0.9 + SPACING * k - SPEED * self.t                       # left edge of pipe k
    def next_pipe(self):
        for k in range(len(self.gaps)):
            if self.pipe_x(k) + PIPE_W > BIRD_X - BIRD_R: return k
        return len(self.gaps) - 1
    def raw(self):
        k = self.next_pipe(); return self.gaps[k] - self.y, self.pipe_x(k) - BIRD_X, self.vy      # dy (negative = gap above), distance to the pipe, vertical speed
    DY = ["far above", "well above", "above", "slightly above", "level", "slightly below", "below", "well below", "far below"]
    VY = ["rising fast", "rising", "rising slowly", "steady", "falling slowly", "falling", "falling fast"]
    def buckets(self):
        dy, dist, vy = self.raw()
        return int(np.searchsorted([-.25, -.12, -.05, -.015, .015, .05, .12, .25], dy)), int(np.searchsorted([-.02, -.012, -.004, .004, .012, .02], vy))
    def text(self):
        b1, b2 = self.buckets(); return f"The next gap is {self.DY[b1]}. The bird is {self.VY[b2]}."
    def step(self, flap):
        if flap: self.vy = FLAP_V
        self.vy += G; self.y += self.vy; self.t += 1
        if self.y < 0 or self.y > 1: self.dead = True; return
        for k in range(len(self.gaps)):
            x = self.pipe_x(k)
            if x < BIRD_X + BIRD_R and x + PIPE_W > BIRD_X - BIRD_R and abs(self.y - self.gaps[k]) > HALF_GAP - BIRD_R: self.dead = True; return
        self.passed = sum(self.pipe_x(k) + PIPE_W < BIRD_X - BIRD_R for k in range(len(self.gaps)))

def make_expert(k, off):
    def expert(env): dy, dist, vy = env.raw(); return (env.y + k * vy) > (env.gaps[env.next_pipe()] + off)      # flap when the predicted height is below the gap centre line
    return expert

def run(seed, policy, lat_ms=0.0, cap=CAP):
    """policy(env) -> bool (flap). A decision started at time t is applied at the first tick at or after t + lat_ms; nothing else happens while it is pending."""
    env = Flap(seed); pending = None
    for k in range(cap):
        now = k * TICK_MS; flap = False
        if pending is not None and pending[1] <= now + 1e-9: flap, pending = pending[0], None
        if pending is None:
            act = bool(policy(env))
            if lat_ms <= 0: flap = flap or act
            else: pending = (act, now + lat_ms)
        env.step(flap)
        if env.dead: break
    return env.passed, env.t
