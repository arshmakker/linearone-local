"""Export the exact 2x2 cube model (move permutations, sticker layout) for the browser demo: writes cube_data.js next to this file."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "../games"))
from cube_env import MOVES, PERMS, SOLVED, SLOTS, NORMALS
faces = {}
for f in "URFDLB": faces[f] = sorted((i for i, (p, n) in enumerate(SLOTS) if n == NORMALS[f]), key=lambda i: SLOTS[i][0])
open(os.path.join(HERE, "cube_data.js"), "w").write("const CUBE = " + json.dumps({"moves": MOVES, "perms": {m: PERMS[m] for m in MOVES}, "solved": list(SOLVED), "faces": faces}) + ";\n")
print("wrote cube_data.js")
