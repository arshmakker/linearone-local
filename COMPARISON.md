# L1 vs Jev vs LLMs: what we measured, by name

All numbers are ours, from small single runs with the caveats shown. **Jev** is the hosted decision API from TypeSafe (`jev-latest`, zero-shot unless stated, our own untuned prompts). **Opus** is Claude Opus, run through Claude Code subagents (batched answers, not independent API calls; no latency or cost measured). **Local LLMs** are llama3.2:3b, qwen2.5:1.5b and 0.5b on an 8-vCPU cloud box. **L1** is the model in this repo. A trained decision tree or embeddings + logistic regression ("simple baselines") is included because in our tests it often beats L1 itself.

## Where L1 wins: speed and cost
| Task | L1 | Jev | Local LLMs |
|---|---|---|---|
| Decision latency (median) | 9-19 ms, local | ~400-430 ms from one laptop | 0.18-2.5 s |
| Snake, live, 40 s | 127 food, ~24 decisions/s, $0 | 15 food, ~2.3 decisions/s, ~$0.002 | 0-2 food (zero-shot) |
| Flap (real-time), hosted with the *right* answers | n/a (L1's own rule scored 0-2 pipes) | **0 pipes at real latency**, 9 with latency removed | not run |
Jev's cost was about $0.000016 per short call; a hosted call also needs a network.

## Where L1 loses (say this out loud)
| Task | Result |
|---|---|
| Snake, 44 unseen states | Opus zero-shot 0.80, Opus with 25 examples **1.000**, Jev zero-shot 0.64, **L1 0.41-0.46** (chance-ish). L1 does not beat Opus or Jev on accuracy here. |
| Compliance mapping, NIST-CSF (200 rules) | Opus 0.83, Jev 0.755, TF-IDF 0.705, **L1 0.68**. |
| Compliance mapping, HIPAA (200 rules) | TF-IDF 0.83, L1 0.76, Jev 0.705, Opus 0.69 (L1 vs Opus not significant, p=0.076). |
| Flap accuracy at 0 ms | decision tree and embeddings + LR 9 pipes; **L1 0-2**. |
| 2x2 cube, optimal-move rate (300 states) | coin flip 18%, Jev 18-24%, L1 21%, decision tree / kNN on raw stickers 50-63%. Nobody solves it except the exact solver. |

## Earlier head-to-heads against Jev (2026-09-23, an older private build; not reproducible from this repo)
L1 trained on task labels vs Jev zero-shot: TweetEval sentiment 68.5% vs 65.8% (12,284 test items); Banking77 93.2% vs 79.7% (3,073); jailbreak detection 97.7% vs 97.2% (a tie, p=0.77). Those show a trained small model beating a zero-shot hosted one where labels are house-specific; they are not evidence about the games above.

## What this supports for a pitch
- **Supported:** a local model answers in real-time loops where a network round trip does not fit; corrections take effect instantly without retraining or a prompt that grows.
- **Not supported:** "L1 is more accurate than Jev or LLMs". On Snake and NIST-CSF, Opus is clearly better; simple baselines match or beat L1.
- Fair framing: *local = latency, cost, privacy, offline; frontier LLM = accuracy on clear-text tasks; a cascade (L1 first, LLM for the uncertain rest) saved 40-50% of Opus calls on NIST-CSF at 0.82 vs 0.83.*

## Caveats to keep with every number
Jev and Opus got one untuned prompt; L1 and baselines were trained on task data. One run per task. Jev latency includes the network from one laptop. Toy games have few distinct states. Opus agents saw many items per prompt.

## Terms
Checked 2026-09-26: TypeSafe's public Terms (updated 2026-09-19), Acceptable Use Policy (updated 2026-09-23) and Privacy Policy contain no clause on benchmarking, publishing comparisons or use of outputs; the Terms prohibit reverse engineering the site's software, which we did not do (black-box API calls only). The logged-in console could not be read (HTTP 403), and no separate API terms were found. This is not legal advice.

## Which models played in each game
| Game | Players |
|---|---|
| **Snake** | L1; Jev (`jev-latest`), zero-shot and with 10 / 25 examples in the prompt; Claude Opus (44 unseen states, zero-shot and with 10 / 25 examples); local LLMs llama3.2:3b, qwen2.5:1.5b, qwen2.5:0.5b (zero-shot); baselines: word TF-IDF + LR, bge-small + LR, decision tree |
| **Flap** | L1 (gamma 100 and 10000); Jev with 0 / 25 / 100 corrections, cached per state, latency injected at 430 ms; baselines: decision tree, word TF-IDF + LR, bge-small + LR; coin flip; scripted expert |
| **2x2 cube** | L1; Jev zero-shot and with 25 examples; baselines: decision tree, kNN on stickers, word TF-IDF + LR, bge-small + LR; coin flip; exact search solver (ground truth). No Opus or local LLMs on this game |
