# l1-local: a small local decision model in C, compared with a hosted model on speed

`l1score` is a single C program (ONNX Runtime C API, no Python at run time) that reads a short text describing a situation and picks one of several options. A frozen small encoder (int8 bge-small-en-v1.5) embeds the text, a scorer picks the option, and you can **teach it corrections on the fly** (`TEACH<TAB>option<TAB>text`, `FORGET`) with no retraining. It runs on one CPU thread in about 10-20 ms and about 90 MB of RAM. This repo also contains three small games that compare it with simple baselines and with a hosted model (Jev, optional, paid, key from `TYPESAFE_API_KEY`).

## What the results say (read this before the demo clip)
- **Speed is the real advantage.** In real-time Flap the hosted model (about 430 ms per decision from a laptop) scores 0 pipes at its real latency, even with correct answers for every state; with latency removed it plays as well as the local models.
- **Accuracy is not.** On all three games a plain decision tree or a frozen encoder + logistic regression on the same data does as well as, or much better than, L1's nearest-example rule. On the Rubik's cube nothing here except a search solver reliably wins, and the hosted model is at chance (18-24% optimal moves).
- The games are toys (a few dozen state texts), so a few lines of code solve them; they show speed and instant correction, not that ML is needed.
- Numbers are from one run each with held-out seeds; `games/PLAN.md` holds the pre-registration and every amendment (including two bugs found before scoring). Results are sensitive to which corrections the learner is given: rerunning with the public int8 file (below) and the default settings gives the same picture but not identical numbers.

## Layout
- `snake/c/l1score.c`, `Makefile`: the runtime. `snake/`: Snake demo (`snake_versus.py` plays L1 and Jev side by side), baselines, corrections.
- `games/`: Flap (real-time) and 2x2 cube benchmarks, `PLAN.md`, result files `*_results*.json` and run logs `*_run*.txt` (these are the numbers we report, from the encoder file we quantised ourselves).

## Build and run
1. Get the ONNX Runtime C library (1.x, CPU) and unpack it to `vendor/onnxruntime` (with `include/` and `lib/`) or pass `ORT=/path`.
2. `make -C snake/c`
3. `./fetch_assets.sh` (downloads Xenova/bge-small-en-v1.5 int8, builds `snake.bin`, `flap.bin`, `cube.bin`; needs python3 with numpy, onnxruntime, tokenizers, scikit-learn).
4. `python3 snake/snake_versus.py --no-jev`, `python3 games/flap_bench.py`, `python3 games/cube_bench.py`. Add `--jev` (and `TYPESAFE_API_KEY`) to include the hosted model; Jev answers are cached locally and not committed.
`DEFAULT_GAMMA` (nearest-example weight, default 100) can be set at build time with `CFLAGS=... -DDEFAULT_GAMMA=10000.0f`; the Flap results use 10000 for the tuned arm.

## Limits
Latencies in Flap and cube time-to-solve are injected from measured values, not measured inside a live game. The Jev latency was measured from one laptop over the network. The bge-small weights are MIT-licensed and are downloaded, not redistributed here. Licence for this repo: to be chosen by the owner.
