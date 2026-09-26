# Reproduce

1. Install the ONNX Runtime C library (CPU, 1.x, from the official releases) into `vendor/onnxruntime` (with `include/` and `lib/`), or pass `ORT=/path` to make.
2. `pip install -r requirements.txt`
3. `make -C snake/c` builds `snake/c/l1score`.
4. `./fetch_assets.sh` downloads the int8 encoder (Xenova/bge-small-en-v1.5, MIT) and builds `snake.bin`, `flap.bin`, `cube.bin`.
5. Run: `python3 snake/snake_versus.py --no-jev` (Snake, L1 only), `python3 games/flap_bench.py`, `python3 games/cube_bench.py`.
The hosted-model arms are optional and paid: set `TYPESAFE_API_KEY` and add `--jev`. Hosted answers are cached locally and are not part of this repo.
The tuned nearest-example weight is a build option: `make -C snake/c CFLAGS="-O3 -DDEFAULT_GAMMA=10000.0f" && mv snake/c/l1score snake/c/l1score_g10000`, then `L1BIN=$PWD/snake/c/l1score_g10000 python3 games/flap_bench.py` (cube: `G10000=...`); rebuild the default afterwards.

## Words used
**L1**: this project's model. **Jev**: a hosted decision API (TypeSafe), used as the "cloud" comparison. **K / N**: number of expert corrections (Flap) / labelled training states (cube). **gamma**: how strongly a taught example pulls the answer. **Nearest-example rule**: answer like the closest taught example. **bge**: the small text-embedding model.

## Limits
Latencies in Flap and cube time-to-solve are injected from measured values, not measured inside a live game. Hosted latency was measured from one laptop. The encoder file you download is a public int8 quantisation and differs slightly from the one behind the committed numbers, so reruns give the same picture but not identical figures (Flap especially depends on which corrections the learner is given). Apache-2.0 licensed (see `LICENSE` and `NOTICE`); bge-small weights are downloaded, not redistributed.

`games/PLAN.md` is a dated, append-only pre-registration (what we would measure and expect, written before running); amendments below it override earlier sections. Two bugs found before scoring are documented there. Each game was run once.
