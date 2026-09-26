"""Render l1_vs_hosted_flap.gif: the Flap game (games/flap_env.py) played twice with the identical expert answer, answered in 12 ms (local) vs 430 ms (hosted).
Same physics and latency rule as the benchmark; only the drawing is added."""
import os, sys
from PIL import Image, ImageDraw, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "../games"))
from flap_env import Flap, make_expert, TICK_MS, HALF_GAP, PIPE_W, BIRD_X as BX, BIRD_R as BR
PW_, PH_ = 300, 260; expert = make_expert(3, 0.05)
try: font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 15); big = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 26)
except OSError: font = big = ImageFont.load_default()
class Player:
    def __init__(s, lat, title): s.lat, s.title, s.seed, s.best, s.crash, s.hold = lat, title, 1000, 0, 0, 0; s.new()
    def new(s): s.env = Flap(s.seed); s.pend = None
    def tick(s):
        e = s.env
        if e.dead:
            s.hold += 1
            if s.hold > 12: s.hold = 0; s.seed += 1; s.crash += 1; s.new()
            return
        now = e.t * TICK_MS; flap = False
        if s.pend and s.pend[1] <= now + 1e-9: flap, s.pend = s.pend[0], None
        if s.pend is None:
            a = bool(expert(e))
            if s.lat <= 0: flap = flap or a
            else: s.pend = (a, now + s.lat)
        e.step(flap); s.best = max(s.best, e.passed)
        if e.t >= 900: s.seed += 1; s.new()
    def panel(s):
        im = Image.new("RGB", (PW_, PH_ + 60), (14, 17, 24)); d = ImageDraw.Draw(im); e = s.env
        for k in range(len(e.gaps)):
            x = e.pipe_x(k)
            if x > 1.05 or x + PIPE_W < 0: continue
            gt, gb = (e.gaps[k] - HALF_GAP) * PH_, (e.gaps[k] + HALF_GAP) * PH_
            d.rectangle([x * PW_, 0, (x + PIPE_W) * PW_, gt], fill=(58, 143, 90)); d.rectangle([x * PW_, gb, (x + PIPE_W) * PW_, PH_], fill=(58, 143, 90))
        c = (255, 107, 107) if e.dead else (255, 209, 102); r = BR * PH_ * 1.6; d.ellipse([BX * PW_ - r, e.y * PH_ - r, BX * PW_ + r, e.y * PH_ + r], fill=c)
        d.text((8, PH_ + 6), s.title, font=font, fill=(230, 233, 240)); d.text((8, PH_ + 26), f"pipes {e.passed}", font=big, fill=(111, 220, 140) if s.lat < 100 else (255, 107, 107))
        d.text((PW_ - 120, PH_ + 34), f"crashes {s.crash}", font=font, fill=(150, 158, 176)); return im
a, b = Player(12, "Local, 12 ms"), Player(430, "Hosted, 430 ms")
frames = []
for t in range(600):
    a.tick(); b.tick()
    if t % 3 == 0:
        im = Image.new("RGB", (PW_ * 2 + 12, PH_ + 60), (14, 17, 24)); im.paste(a.panel(), (0, 0)); im.paste(b.panel(), (PW_ + 12, 0)); frames.append(im)
frames[0].save(os.path.join(HERE, "l1_vs_hosted_flap.gif"), save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True)
print(len(frames), "frames")
