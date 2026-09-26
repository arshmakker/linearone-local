# Three games for the open-source demo (plan and pre-registration, 2026-09-26)

Purpose: show what a small local decision model does, and does not do, against hosted models and simple classifiers. Every game reports the coin flip, the scripted expert (ceiling), and the same baselines; results are published whatever they are.
Game 1 is Snake (snake/, done). Lessons carried over: (a) also report a decision tree on the underlying numbers and frozen encoder + scaled logistic regression, because they match or beat L1's nearest-example rule;
(b) give hosted models the same corrections as the local learners; (c) state that a game a few lines of code solve shows speed and instant correction, not that ML is needed.

## Game 2: Flap (real-time, latency is the point). Pre-registered before any result.
World (fixed before evaluating any learner): 30 ticks per second; y down, world height 1.0; bird at x = 0.25; gravity 0.0016 per tick; a flap sets vy = -0.021; pipe speed 0.008 per tick; a pipe every 90 ticks; gap half-height 0.17; gap centre uniform in [0.30, 0.70] by seed; pipe
width 0.08; dies on a pipe or the floor/ceiling; episode cap 900 ticks (30 s). Score = pipes passed; also survival ticks. State text (what the model sees): "The next gap is {far above|above|level|below|far below}. The pipe is {far|near|very near}. The bird is {rising|steady|falling}." (45 states);
the same three buckets as integers feed the decision tree. Scripted expert = a fixed rule on the raw numbers (constants frozen before running any learner).
Latency semantics: a policy answers one query at a time. A decision made on the state at time t takes L ms and is applied at the first tick at or after t+L; the bird keeps flying (no flap) meanwhile. Arms are run at L = 0 (accuracy alone) AND at their real median latency:
L1 12 ms (C binary, measured), TF-IDF 1 ms, frozen bge-small + scaled LR 4 ms, decision tree 0.1 ms, Jev 430 ms (measured earlier from this laptop), coin flip 0.
Learners get the first K = 25 and 100 corrections of the scripted expert, collected in zero-latency teaching games (the model flies, errors are corrected and counted). Arms: coin flip (flap with probability 0.5 each decision), always no-flap, scripted expert, decision tree on the 3 bucket features, word TF-IDF + LR,
frozen bge-small (int8) + StandardScaler + LR, L1 nearest taught state, Jev zero-shot and Jev with the same K corrections in the prompt (paid, opt-in; answers cached per distinct state text, so about 45 calls per configuration).
Evaluation seeds 1000-1019 (20 games), never used for teaching. One run. Metrics: mean pipes passed, mean survival ticks, share of games reaching the cap. Expectation stated in advance: the local learners will play at L = 0 roughly in line with each other (the tree the best); Jev at 430 ms will fail at the first pipe;
the point of the game is the latency column, not the accuracy column.
Limits stated in advance: the bucketed state has only 45 distinct texts, so every policy is effectively a lookup table; latency here is injected (simulated) from measured values, not measured inside a live game; Jev answers are cached per state text.

## Game 3: Rubik's cube (2x2x2 pocket cube). Pre-registered 2026-09-26, before any code or result.
World: corner DBL is fixed (3,674,160 states); moves U, U', R, R', F, F' (quarter turns). Ground truth: breadth-first search from the solved state to depth 7 gives the exact distance for every state within 7 moves; a move is optimal if it lands on a state one step closer.
State text (what models see): six lines, one per face (U, R, F, D, L, B), each listing its four stickers as colour words, e.g. "U face: white white red blue."; same info as the 24 one-hot sticker features fed to the decision tree and to kNN.
Task: pick a move; correct = any optimal move. Data: scrambles of depth uniform in 1..6 (random walk without immediate inverse), labelled by the solver with its first optimal move in the fixed order above. Teach seeds 1..; eval set = 300 states from seeds 1000+ (never used for teaching or tuning), depth uniform 1..6; a second test = solve 20 scrambles of depth 4 by applying the model's move greedily, cap 12 moves, solved = success.
Arms: coin flip (uniform over the 6 moves), always "U", decision tree on 24 sticker one-hots, kNN (k=1) on the same, word TF-IDF + LR, frozen bge-small + StandardScaler + LR, L1 nearest-example (taught, gamma 10000 per Game 2, and 100 as shipped), Jev zero-shot and Jev with K=25 examples in the prompt (paid, opt-in; 300 eval calls each). Learners get N = 200 and N = 2000 labelled states (Jev and L1's memory: N = 25 and 200; Jev K=25 only, because prompt size).
Latency (injected from measured values, as in Game 2, added to time-to-solve, not to accuracy): L1 12 ms, TF-IDF 1 ms, bge+LR 4 ms, tree/kNN 0.1 ms, Jev 430 ms.
Metrics: optimal-move rate on the 300 eval states (with Wilson 95% interval), solve rate on the greedy test, mean moves and mean seconds to solve.
Expectations stated in advance: (1) coin flip about 20-25% optimal-move rate (several moves can tie); (2) all text/embedding arms stay below 50% even at N=2000 (colour-word text is a poor cube encoder); (3) the decision tree/kNN on one-hots do best among learners but still solve almost none of the depth-4 scrambles; (4) Jev zero-shot near coin flip; (5) nobody except the BFS solver reliably solves. Point of the game: an honest contrast where a small local model is not enough and neither is a hosted model, and only search solves it.
Limits stated in advance: the solver is a ground-truth oracle (search), not a competitor; one run; a smaller-than-full state space in evaluation (depth <= 6).

## Frozen expert (2026-09-26, before any learner or eval seed)
Scripted expert = flap when `y + 3*vy > gap_centre + 0.05`. Chosen on tuning seeds 1-30 only (grid k in 3..12, offset -0.04..0.06; only k=3 with offset 0.04-0.06 reached the 900-tick cap on all 30). Offset 0.05 sits between the two passing grid points. Constants are now frozen; eval seeds 1000-1019 have not been touched.

## Amendment 1 (2026-09-26, before any scored result): state text made finer
Finding: under the pre-registered 45-text state, the best possible lookup table (majority vote of the expert over 200 games) scored 0 pipes (the state is too coarse for any policy), and all eight local learner arms also scored 0. That result was informationless, so the state was changed before any comparison that could count.
Disclosure: that bound check flew eval seeds 1000-1019 once (0 pipes); the new bucket edges were chosen on seeds 1-200 and scored on tuning seeds 201-230 only.
New state text (63 states; the pipe-distance clause is dropped because it did not help): "The next gap is {far above|well above|above|slightly above|level|slightly below|below|well below|far below}. The bird is {rising fast|rising|rising slowly|steady|falling slowly|falling|falling fast}."
dy edges (gap centre minus bird height) -0.25,-0.12,-0.05,-0.015,0.015,0.05,0.12,0.25; vy edges -0.02,-0.012,-0.004,0.004,0.012,0.02. Majority-vote lookup on the new state: 9.0 pipes, 100% cap on tuning seeds 201-230. Everything else in this plan is unchanged; the decision tree now takes these two integer features.
Eval seeds 1000-1019 are re-run once under this amendment; the earlier 0-pipe run is reported as the reason for the amendment.

## Amendment 2 (2026-09-26, before the scored eval run): L1 nearest-example weight
Finding on tuning seeds 201-230 (K=100 corrections): the shipped nearest-example weight gamma=100 scores 0.0 pipes (the 63 state texts sit at about 0.99 cosine to each other, so a weight of 100 cannot separate them); gamma=1000 scores 2.5; gamma=10000 scores 2.5 (K=25: 2.6). gamma became a build-time option (`-DDEFAULT_GAMMA`, default unchanged at 100).
Scored run reports L1 at gamma=100 (as shipped) and at gamma=10000 (tuned on 201-230). The scored table also includes Jev zero-shot / K=25 / K=100 (paid, opt-in, answers cached per state text; criteria text in `jev_flap.py`).

## Amendment 3 (2026-09-26, before any Game 3 result): 64-token cap, latencies
The cube text is 50 tokens; the 32-token cap of Games 1-2 would cut off the D, L and B faces for the encoder arms, so Game 3 uses a 64-token cap (cube.bin header and the Python encoder). Latency for L1 is therefore assumed 20 ms (to be replaced by the measured median printed by the run) and bge+LR 8 ms in the time-to-solve column; accuracy is unaffected by these numbers. The Jev greedy-solve test calls Jev live on unseen states (cached per state text).

## Amendment 4 (2026-09-26): leak found and fixed before reporting any Game 3 number
The first Game 3 run used teaching seeds 1..2000, which contain the eval seeds 1000..1299 (identical scrambles); tree, kNN and bge+LR scored 100% for that reason. Discarded. Teaching seeds are now 100000+ and any state that appears in the eval or solve sets is removed from the teaching data, so eval states are never taught (at depth <= 3 all states are few, so some depth-1..3 states are excluded from teaching entirely).
