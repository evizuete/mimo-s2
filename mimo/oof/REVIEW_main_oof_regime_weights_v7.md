# Code review — `mimo/oof/main_oof_regime_weights_v7.py`

Scope: the OOF metamodel entrypoint and its immediate collaborators
(`OptunaOOFTrainer.evaluate_holdout*`, `DataPipeline.create_sequences_by_side`,
`Helper.build_nonzero_final_weights`).

The findings below are ordered by severity. Line numbers reference the file at
HEAD on `claude/review-oof-metamodel-TnEFr` (1389 lines).

---

## Critical

### C1. Database credentials hard-coded in source
`mimo/oof/main_oof_regime_weights_v7.py:1073`

```python
optuna_db="mysql+pymysql://evizuete:Ev1z43t3.00@10.1.21.25:3306/optuna_db",
```

Plain-text password and host committed to the repo. Rotate the credential and
read it from an environment variable (`os.environ["OPTUNA_DB_URL"]`) or a
secrets manager. This blocks any external publication of the repository.

### C2. New release entries 200401/200402/200403 will crash `build_trainer`
`main_oof_regime_weights_v7.py:543-555` (added in commit `b703435`)

```python
"200401": {"tp_base": 1.00, "sl_base": 0.80},
"200402": {"tp_base": 1.25, "sl_base": 0.80},
"200403": {"tp_base": 1.00, "sl_base": 1.00},
```

These entries omit `regime_barriers_long` and `regime_barriers_short`. The
caller at `main_oof_regime_weights_v7.py:1053-1055` does

```python
regime_barriers_long=barriers["regime_barriers_long"],
...
regime_barriers_short=barriers["regime_barriers_short"],
```

so any of the three releases will raise `KeyError: 'regime_barriers_long'`
the first time `build_trainer` runs. Plus, `_get_barriers_for_release`
prints `barriers['regime_barriers_long']['trending']` at line 555, which
raises before `build_trainer` even gets called. Either:

- backfill `regime_barriers_long/short` for each new entry (copying from
  200398 if no per-state tuning is desired), or
- treat missing keys as "use the bases for every regime" by post-processing
  the dict inside `_get_barriers_for_release`.

### C3. Hard-coded train/holdout date windows
`main_oof_regime_weights_v7.py:1281-1283`

```python
optuna_from  = datetime(2025, 1, 1)
holdout_from = datetime(2026, 2, 1)
holdout_to   = datetime(2026, 4, 26)
```

Every release is forced to the same window. Promoting these to CLI flags
(already done on `multitask` in commit `3915cd0`) is a one-line fix and is
required for backfills, walk-forward sensitivity tests, and reproducibility of
older releases. While they remain hard-coded, `--release` is misleading because
the *data* used is independent of the release tag.

---

## High

### H1. Closure capture in `install_regime_weight_patch` is silently sticky
`main_oof_regime_weights_v7.py:901-1025`

`_PATCH_INSTALLED` short-circuits on the second call (`if _PATCH_INSTALLED:
return`), so any subsequent invocation with a *different* `regime_weights_by_side`
is silently ignored — the originally captured map is reused. In a one-shot
script this is fine, but it is a footgun for tests, notebooks, or any future
caller that wants to sweep variants in-process. Either:

- store `regime_weights_by_side` in a module-level mutable that the wrapper
  reads on every call, or
- raise on re-install with a different map.

### H2. Global mutable state for audit
`main_oof_regime_weights_v7.py:621, 1000-1008, 1237, 1247, 1369`

`_LAST_AUDIT` is a module-level dict written by the monkey-patch and read by
`run_side` and the combined report. Side ordering and any partial failure
between the two reads can produce a silently stale or cross-contaminated audit
in the persisted JSON. Pass the audit explicitly through the call chain (the
trainer already returns artifacts; piggy-back on those), or scope it per
`run_side` invocation.

### H3. Alignment of `target_df` with sequence labels relies on an undocumented invariant
`main_oof_regime_weights_v7.py:946-960`

```python
target_df = df.iloc[-n:].copy()
states = target_df["state"].astype(str)
labels_arr = np.asarray(seq["labels"], dtype=np.int32)
base_weights = np.asarray(seq["weights"], dtype=np.float32)
```

This is correct *today* because
`DataPipeline.create_sequences_by_side` slices labels as
`df['signal'].values[context_offset_base:context_offset_base + n_samples]` with
`n_samples = len(df) - (seq_len_long - 1)`. Any future change that drops tail
rows, applies a horizon-dependent mask, or adjusts `n_samples_full` (see
`data_pipeline_v2.py:147`) will silently misalign weights with labels and the
training will be poisoned without raising.

Defensive fix: have `create_sequences_by_side` return the row indices (or
timestamps) it kept and have the patch index `df` with those indices instead
of trusting `df.iloc[-n:]`.

### H4. Holdout evaluation calls `create_sequences_by_side(..., train=True)`
`mimo/oof/optuna_oof_trainer_v2.py:745-748` and `:798`

The comment in v7 documents why the monkey-patch is bypassed via
`_HOLDOUT_EVAL_ACTIVE` (lines 624-647). The bypass is correct, but the design
is fragile — it depends on every future caller of `evaluate_holdout*` being
wrapped in `holdout_eval_context()`. Two cleaner options:

1. Make `create_sequences_by_side` accept a `regenerate_labels_only=True` flag
   that returns labels without populating `weights`, removing the need for the
   monkey-patch in eval paths altogether.
2. Detect eval intent inside the wrapper itself (e.g. via a `purpose` kwarg
   the trainer sets), so future trainers can't accidentally re-introduce the
   bug.

Today the bypass works; tomorrow nobody will remember why.

---

## Medium

### M1. Duplicated implementation of `build_nonzero_final_weights`
`main_oof_regime_weights_v7.py:691-792` vs `mimo/helpers/helper.py:110-260`

`_local_build_nonzero_final_weights` is a near line-by-line copy of
`Helper.build_nonzero_final_weights`. The only differences relevant here are
default values (the helper uses `min_base_weight=0.10`, the local fallback is
called with `min_base_weight=0.0` from `_build_final_weights_for_side`).

The fallback exists because `Helper.build_nonzero_final_weights` may not be
"invocable tal cual" (per the file docstring, line 11). If that constraint is
still real, please document *which* environments trigger it; otherwise drop
the fallback and call the helper directly. Two implementations of weight
construction is a maintenance bug waiting to happen.

### M2. `feature_masks` are *additive*, not exclusive
`main_oof_regime_weights_v7.py:1058-1061`

```python
feature_masks={
    "long":  {"ema_bull": True, "rsi_oversold": True, "macd_positive": True},
    "short": {"ema_bear": True, "rsi_overbought": True, "macd_negative": True},
},
```

On `multitask` (commit `be933be`) the semantics were flipped to *exclusive*.
Anyone who reads this v7 file expecting the new semantics will train against
a wider feature set than intended — silently. Even on this branch, document
the active mask semantics in the FeatureConfig docstring and assert which
mode is in use.

### M3. `label_horizon = max(h_long, h_short)` masks per-side mismatches
`main_oof_regime_weights_v7.py:1049`

```python
label_horizon=max(label_horizon_long, label_horizon_short),
```

`FeatureConfig` carries a *single* `label_horizon` plus `label_method_long`
and `label_method_short`. If the side-specific labelers honor `label_horizon`
verbatim instead of branching on side, asymmetric horizons (e.g. `h_long=5`,
`h_short=3` from the recent run) produce labels for the *longer* horizon on
the shorter side. Verify the labeler honors per-side horizon and add an
explicit `label_horizon_long` / `label_horizon_short` to `FeatureConfig` if
not.

### M4. `--variant-long` default is `moderate`, but `moderate_h15` is the only variant that resolves to `h=15`
`main_oof_regime_weights_v7.py:585-586`

```python
h_long = 15 if args.variant_long == "moderate_h15" else 10
```

This couples horizon to *variant name*. A user who passes `--variant-long
trend_robustness_v1 --label-horizon-long 15` works, but
`--variant-long moderate_h15 --label-horizon-long 5` is also accepted and
produces a tag (`rw_both_Lmoderate_h15_h5_...`) where the variant name
contradicts the horizon. Either:

- decouple horizon from variant entirely (force `--label-horizon-long`), or
- validate that variant `moderate_h15` is only used with `h=15`.

### M5. `_DEFAULT_BARRIERS` is far from the currently-explored region
`main_oof_regime_weights_v7.py:445-460`

`tp_base=2.5, sl_base=1.5` and the embedded `regime_barriers_long/short`
date back to the 200391-200395 family. Any new release without an entry in
`BARRIERS_BY_RELEASE` will silently regress to this old config.
`_get_barriers_for_release` only emits an info log; it should warn
loudly or refuse to run when no entry is present, given how many recent
releases (200398, 200399, 200400) had specific barrier tuning.

### M6. `combined_report` and per-side `report` JSON files include the full holdout report
`main_oof_regime_weights_v7.py:1234-1244, 1376-1378`

The walk-forward holdout report carries prediction arrays
(`return_predictions=True`, line 1185) which, after `_json_safe`, become
huge nested lists in the JSON file. For the 202103 run that's ~34k rows ×
several columns. Either drop predictions before persisting the report or
write predictions only to parquet and keep JSON metadata-only.

### M7. `out_dir=str(train_dir)` mixes `Path` and `str`
`main_oof_regime_weights_v7.py:1072`

The trainer is constructed with a string and uses `os.path.join` internally,
while the rest of the file uses `pathlib.Path`. Pick one — passing the
`Path` directly works on every Python ≥3.6 stdlib API and removes a class of
"forgot to wrap" bugs.

---

## Low

### L1. `dump_json_safe` is good; please use it everywhere
`Helper.save_meta` (`mimo/helpers/helper.py:68-72`) uses `json.dump(...,
default=str)`, while `main_oof_regime_weights_v7.py` uses `_json_safe`. They
behave differently for `pd.Timestamp` (`default=str` produces
`'2026-04-26 00:00:00'`, `_json_safe` produces `'2026-04-26T00:00:00'`).
Persist consistently to avoid downstream parsing forks.

### L2. `evaluator.evaluate_predictions` thresholds are hard-coded inside the trainer
`mimo/oof/optuna_oof_trainer_v2.py:773-776` and `:826-828`

```python
beta_primary=0.25, min_precision=0.45, max_signal_rate=0.15
```

These should live in a `EvalConfig` (or be CLI args) rather than embedded in
two near-identical methods. The duplication between `evaluate_holdout` and
`evaluate_holdout_v0` doubles the maintenance burden.

### L3. Mixed Spanish/English throughout
The codebase mixes both languages in identifiers, log lines, and docstrings.
Not a correctness issue but worth aligning if outside contributors are
expected.

### L4. `GRID_COMMON = _DEFAULT_GRID` aliased "for retrocompat"
`main_oof_regime_weights_v7.py:436`

If nothing imports it (the comment says "no se usa en el flujo"), delete it
rather than carry dead exports.

### L5. `oof.docx` checked in next to source
`mimo/oof/oof.docx`

Binary documentation living alongside `.py` files makes diffs noisy. Either
move it to a `docs/` folder or convert to markdown.

### L6. `parse_args` defaults `--release` to `"200382"`
`main_oof_regime_weights_v7.py:569`

A default that does not appear in `GRID_BY_RELEASE` or
`BARRIERS_BY_RELEASE` will silently fall back to defaults and warn only via a
single `print`. Consider failing fast (`raise SystemExit`) when no entry
exists, or default to the most recent release explicitly.

### L7. `print` is not a logger
A run of this script produces ~hundreds of lines of `print` output mixed
between the script, the trainer, and the data pipeline. A
`logging.getLogger(__name__)` with a single root config would let users
dial verbosity for diagnosis vs. CI.

---

## Things that look right

- `_json_safe` is exhaustive (handles `Path`, `Timestamp`, `Timedelta`,
  numpy scalars, ndarray, Series, Index, DataFrame, dict, list/tuple/set).
- `holdout_eval_context` is a context manager — it correctly resets
  `_HOLDOUT_EVAL_ACTIVE` even on exception (try/finally, line 643-647).
- `load_existing_artifacts` validates required files before returning
  (lines 1098-1108) — a good fix per the v5 note.
- The renormalization (`renorm_to_base_sum=True`) keeps total mass stable
  so that loss scale is roughly comparable across variants.
- Correct guard `if not train_flag: return results` (line 920) prevents the
  patch from interfering with inference / scaler-only passes.

---

## Suggested fix order

1. Rotate the DB credential and move it to env (C1). Block all other work
   until this is shipped.
2. Backfill missing `regime_barriers_long/short` for 200401/200402/200403
   (C2) — these will crash on first run.
3. Promote `optuna_from / holdout_from / holdout_to` to CLI flags (C3).
4. Add a single source of truth for `build_nonzero_final_weights` (M1).
5. Make `regime_weights_by_side` re-installable, or assert on conflicting
   re-install (H1).
6. Have `create_sequences_by_side` return the row indices it kept and use
   those in the monkey-patch (H3) — this also kills the bypass-flag design
   (H4).
7. Audit `label_horizon` per-side and either pass both or assert equality
   (M3).
8. Cosmetic: dedupe grid/barrier dicts, add logging, drop `oof.docx`.
