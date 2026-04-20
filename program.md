# autoresearch — Heterogeneous Graph Diffusion (HGD)

Apply the autoresearch loop to the HGD project: autonomously experiment with
model architecture, diffusion design, and hyperparameters to maximize graph
classification F1 on MUTAG.

---

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr11`).
   The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current `main`.
3. **Read the in-scope files** for full context:
   - `README.md` — project overview and forward-process math.
   - `src/models/hgd.py` — HGD model (main file you modify).
   - `src/models/denoising_net.py` — Denoising U-Net backbone (may also modify).
   - `src/models/ddm.py` — DDM baseline (read-only reference).
   - `src/evaluate.py` — evaluation harness. **Do not modify.**
   - `src/datasets/data_util.py` — data loading. **Do not modify.**
   - `train_hgd.py` — training entry point. **Do not modify.**
   - `configs/MUTAG.yaml` — hyperparameters. **You may modify this.**
4. **Verify dependencies**: run `pip show dgl torch` to check that DGL and PyTorch
   are available. If not, tell the human to `pip install -r requirements.txt` first.
5. **Initialize results.tsv**: create `results.tsv` with just the header row.
   The baseline will be recorded after the first run.
6. **Confirm and go**: confirm setup looks good.

Once confirmed, kick off experimentation.

---

## Experimentation

Run experiments as:

```
python train_hgd.py --config configs/MUTAG.yaml > run.log 2>&1
```

**What you CAN modify:**
- `src/models/hgd.py` — forward diffusion process, noise schedule, loss, Laplacian
  construction, time embedding, model `__init__`, `forward`, `embed` methods.
- `src/models/denoising_net.py` — U-Net architecture: GAT layers, skip connections,
  MLP blocks, normalization, activation, time injection strategy.
- `configs/MUTAG.yaml` — any hyperparameter: `T`, `beta_schedule`, `beta_1`,
  `beta_T`, `num_hidden`, `num_layers`, `nhead`, `LR`, `MAX_EPOCH`,
  `weight_decay`, `BATCH_SIZE`, `eval_T`, `seeds`, etc.

**What you CANNOT modify:**
- `src/evaluate.py` — this is the ground truth metric.
- `src/datasets/` — data loading is fixed.
- `train_hgd.py` — training entry point is fixed.
- `prepare.py` / `src/utils/` — supporting utilities; treat as read-only.

**The goal**: maximize `mean_f1` (the `Final:` line in the log). Higher is better.
MUTAG is a small dataset (188 graphs, binary classification), so runs are fast —
expect 1–5 minutes per experiment at 100 epochs.

**Simplicity criterion**: All else being equal, simpler is better. A tiny F1 gain
that adds complex code is not worth it. Removing code and matching/beating F1?
Definitely keep. Weigh complexity cost against improvement magnitude.

**The first run**: always establish the baseline — run the training script as-is
with no changes.

---

## Output format

The training script logs to `logs/hgd/train.log`. The final summary line looks like:

```
2026-04-11 12:00:00 INFO Final: 0.8947 ± 0.0312
```

Extract the key metrics:

```bash
grep "Final:" logs/hgd/train.log | tail -1
grep "#Test_f1:" logs/hgd/train.log | tail -5
```

If the grep output is empty, the run crashed. Read the traceback:

```bash
tail -n 60 run.log
```

---

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT
comma-separated — commas break inside descriptions).

The TSV has a header row and 5 columns:

```
commit	mean_f1	std_f1	status	description
```

1. git commit hash (short, 7 chars)
2. mean F1 achieved (e.g. 0.894700) — use 0.000000 for crashes
3. std F1 (e.g. 0.031200) — use 0.000000 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	mean_f1	std_f1	status	description
a1b2c3d	0.894700	0.031200	keep	baseline
b2c3d4e	0.902100	0.028500	keep	increase LR to 0.001
c3d4e5f	0.881000	0.040000	discard	switch to linear beta schedule
d4e5f6g	0.000000	0.000000	crash	double num_hidden (OOM)
```

---

## The experiment loop

The experiment runs on branch `autoresearch/<tag>`.

LOOP FOREVER:

1. Look at git state: the current branch/commit.
2. Propose and apply an experimental idea — edit `src/models/hgd.py`,
   `src/models/denoising_net.py`, and/or `configs/MUTAG.yaml`.
3. `git commit` the change.
4. Run: `python train_hgd.py --config configs/MUTAG.yaml > run.log 2>&1`
5. Read results: `grep "Final:" logs/hgd/train.log | tail -1`
6. If grep is empty → crashed. Read `tail -n 60 run.log`. Attempt a fix if trivial;
   otherwise mark as crash, log it, and `git reset --hard HEAD~1`.
7. Record results in `results.tsv` (NOT committed — leave it untracked).
8. If mean_f1 **improved** → advance the branch (keep the commit).
9. If mean_f1 is equal or worse → `git reset --hard HEAD~1` (discard the commit).

**Timeouts**: If a run exceeds 15 minutes, kill it (`Ctrl-C`) and treat as a crash.

**Crashes**: Fix trivial bugs (typo, missing import) and re-run. If the idea is
fundamentally broken, skip it, log "crash", and move on.

**NEVER STOP**: Once the loop has begun, do NOT pause to ask the human if you
should continue. Do NOT ask "should I keep going?". The human may be asleep.
You are autonomous. If you run out of ideas, think harder — explore alternative
noise schedules, Laplacian variants, attention mechanisms, pooling strategies,
loss functions, data augmentation, multi-scale embeddings, positional encodings.
The loop runs until the human interrupts you, period.

---

## Research directions to explore

Below are suggested starting points (not exhaustive). Pick the most promising one
given the current state of the experiment log.

**Noise schedule**:
- Try `linear`, `quad`, `jsd`, `const` beta schedules.
- Tune `beta_1` and `beta_T` values.
- Extend or shorten `T` (number of diffusion steps).

**Laplacian dynamics**:
- Try a symmetric normalized Laplacian instead of the current `D - A/n`.
- Clamp or normalize `diffused_x` to avoid exponential blow-up at large `t`.
- Use a random-walk Laplacian.

**Architecture**:
- Increase/decrease `num_hidden`, `num_layers`, `nhead`.
- Replace GAT with GCN or GraphSAGE layers.
- Add residual connections across diffusion timestep embeddings.
- Try sinusoidal time embeddings instead of `nn.Embedding`.
- Use multi-head pooling (mean + max concatenation) instead of mean-only.

**Training**:
- Tune `LR`, `weight_decay`, `MAX_EPOCH`, `BATCH_SIZE`.
- Try cosine annealing or warmup LR schedules.
- Try Adam instead of AdamW.

**Loss**:
- Tune `alpha_l` in `sce_loss`.
- Try MSE loss instead of SCE.
- Add a contrastive auxiliary loss.

**Evaluation**:
- Extend `eval_T` to more timestep values.
- Run more seeds for a more stable estimate (edit `seeds` in config).
