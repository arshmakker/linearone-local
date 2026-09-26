"""Render a recorded snake_versus run (clip_frames.jsonl) to an MP4 with PIL + ffmpeg. The frames are the screen text of a real live run;
this script only draws them (colours for the board, a title, and the caveats as a caption). Nothing is simulated."""
import json, os, subprocess, sys
from PIL import Image, ImageDraw, ImageFont

src = sys.argv[1] if len(sys.argv) > 1 else "clip_frames.jsonl"; out = sys.argv[2] if len(sys.argv) > 2 else "l1_vs_jev_snake.gif"
FPS, SIZE = 8, 16
frames = [json.loads(l) for l in open(src)]
font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", SIZE); bold = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", SIZE, index=1)
CAP = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 11)
cw = font.getlength("M"); lh = int(SIZE * 1.42); cols = max(len(l) for f in frames[:5] for l in f["screen"].split("\n")) + 2
CAPTION = ["L1: local C binary, taught by 25 corrections from a scripted expert, then plays alone. Jev: hosted API, zero-shot, no examples.",
           "Toy task (Snake): shows speed and learning from corrections, not compliance accuracy.",
           "Jev prompt was one untuned attempt; Jev latency includes the network. L1 latency here includes Python overhead (standalone C: about 9-12 ms)."]
W, H = int(cw * (cols + 2)), lh * (len(frames[0]["screen"].split("\n")) + 3)
BG, DIM, WHITE, GREEN, ORANGE, RED, CYAN = (18, 20, 26), (95, 100, 112), (225, 228, 235), (110, 220, 140), (255, 170, 80), (255, 95, 95), (90, 200, 255)
def colour_for(ch): return {"@": CYAN, "o": GREEN, "*": RED, ".": DIM, "|": DIM}.get(ch, WHITE)
def draw_frame(screen):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im); y = lh // 2
    d.text((cw, y), "L1 vs Jev: the same Snake games, played live (recorded 2026-09-26)", font=bold, fill=WHITE); y += int(lh * 1.4)
    for line in screen.split("\n")[2:]:
        board = line.strip() and set(line) <= set(". @o*|") and "." in line
        if board:
            x = cw
            for ch in line: d.text((x, y), ch, font=bold if ch in "@*" else font, fill=colour_for(ch)); x += cw
        else:
            fill = GREEN if line.startswith("L1") or (line.startswith("food") and False) else WHITE
            if line.startswith("L1  (") or "  |  Jev" in line and line.startswith("L1"):
                left, _, right = line.partition("|"); d.text((cw, y), left, font=bold, fill=GREEN); d.text((cw + cw * len(left), y), "|" + right, font=bold, fill=ORANGE)
            else: d.text((cw, y), line, font=font, fill=fill)
        y += lh
    y += lh // 2
    for c in CAPTION: d.text((cw, y), c, font=font if False else CAP, fill=DIM); y += int(lh * 0.8)
    return im
end = frames[-1]["t"]; i = 0; n = int(end * FPS); imgs = []
for k in range(n):
    t = k / FPS
    while i + 1 < len(frames) and frames[i + 1]["t"] <= t: i += 1
    imgs.append(draw_frame(frames[i]["screen"]).convert("P", palette=Image.ADAPTIVE, colors=32))
imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=int(1000 / FPS), loop=0, disposal=1)
print(f"wrote {out}: {n} frames at {FPS} fps, {end:.0f} s, {W}x{H}, {os.path.getsize(out)/1e6:.1f} MB")
