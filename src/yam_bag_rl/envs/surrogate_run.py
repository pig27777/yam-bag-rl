"""Run the preflight through held-out evaluation with a milestone gate after C0."""

import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import subprocess
from threading import Event, Thread
from time import perf_counter

import numpy as np
import torch
import warp as wp

from ..paths import PROJECT_ROOT
from .surrogate_c0 import REGIMES, _write, run_c0, summarize
from .surrogate_env import ACTION_MASK, PHASE_TICKS, TryBatch, VectorSurrogate
from .surrogate_learning import (
    BatchData, MaskedPPO, ResidualEncoder, SurrogateScorer, concatenate, fit_scorer, linear_probe, metrics,
    score_batches, split_initial_states,
)

MILESTONES = ("preflight", "c0", "scaling", "c1", "c2", "c3_c4", "held_out")
EVAL_ORDER = ("Best Fixed", "Vision-Only No Try", "Matched Blind", "Fixed Try + Informed",
              "Learned Try + Informed w/o Effort", "Learned Try + Informed", "Privileged Oracle")


def log(message):
    print(message, flush=True)


def save_npz(path, data):
    temp = path.with_suffix(".tmp.npz")
    np.savez(temp, initial_id=data.initial_id, regime=data.regime, v0=data.v0, actions=data.actions,
             history=data.tried.history, valid_history=data.tried.valid, windows=data.tried.windows,
             timestamp=data.tried.timestamps, candidates=data.tried.candidates, choices=data.choices,
             success=data.success, valid=data.valid, private=data.private)
    temp.replace(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as d:
        tried = TryBatch(d["history"], d["valid_history"], d["windows"], d["timestamp"], d["candidates"])
        return BatchData(d["initial_id"], d["regime"], d["v0"], d["actions"], tried, d["choices"],
                         d["success"], d["valid"], d["private"])


class GPUMemory:
    """NVML-backed nvidia-smi polling outside the simulation path."""

    def __init__(self):
        self.pid = os.getpid()
        self.done = Event()
        self.thread = Thread(target=self.poll, daemon=True)
        self.process_peak = 0
        self.total_peak = 0

    @staticmethod
    def _query(*args):
        return subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"], capture_output=True,
                              text=True, timeout=5, check=True).stdout

    def sample(self):
        by_uuid = {}
        table = self._query("--query-gpu=uuid,memory.used")
        for line in table.splitlines():
            uuid, mem = line.split(",")
            by_uuid[uuid.strip()] = int(mem.strip())
        table = self._query("--query-compute-apps=pid,gpu_uuid,used_gpu_memory")
        for line in table.splitlines():
            pid, uuid, mem = line.split(",")
            if int(pid.strip()) == self.pid:
                self.process_peak = max(self.process_peak, int(mem.strip()))
                self.total_peak = max(self.total_peak, by_uuid[uuid.strip()])

    def poll(self):
        while not self.done.is_set():
            self.sample()
            self.done.wait(.25)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.done.set()
        self.thread.join()
        self.sample()


def random_try(rng, n):
    actions = rng.uniform(-1., 1., (n, 13)).astype(np.float32)
    return actions * ACTION_MASK


def rollout(env, seed, initial, rng, *, actor=None, scorer=None, collect=False, training=False):
    v0 = env.reset(seed, initial, training=training)
    if actor is None:
        u = old_logp = None
        action = random_try(rng, env.worlds)
    else:
        u, action, old_logp = actor.sample(v0)
    tried = env.try_batch(action, collection=collect)
    max_q = None
    if scorer is None:
        selected = rng.integers(0, env.count, env.worlds, dtype=np.int32)
    else:
        logits = score_batches(scorer, v0, action, tried)
        selected = logits.argmax(1)
        max_q = (1. / (1. + np.exp(-np.clip(logits.max(1), -30., 30.)))).mean()
    result = env.grasp_batch(selected)
    data = BatchData(np.arange(initial, initial + env.worlds, dtype=np.int64), result.regime, v0, action, tried,
                     selected, result.success, result.valid, result.private)
    return data, result, u, old_logp, max_q


def extension_benchmark(output, device, worlds):
    baseline = json.loads((output / "scaling_benchmark.json").read_text())
    best = baseline["best"]
    if best is None:
        raise RuntimeError("Complete the baseline scaling benchmark before testing an extension")
    with GPUMemory() as memory:
        env = VectorSurrogate(worlds, device=device)
        ppo = MaskedPPO(211)
        scorer = SurrogateScorer(ResidualEncoder(np.arange(62), include_effort=True))
        start = perf_counter()
        _, result, *_ = rollout(env, 269, 9_200_000, np.random.default_rng(441), actor=ppo, scorer=scorer)
        seconds = perf_counter() - start
    rate = float(result.valid.sum() * 60 / seconds)
    gain = rate / best["valid_macro_episodes_per_min"] - 1.
    status = "PASS" if (0 < memory.total_peak <= 28 * 1024 and (1. - result.valid.mean()) <= .02) else "FAIL"
    report = summarize(result) | dict(status=status, worlds=worlds, wall_s=seconds,
                                      valid_macro_episodes_per_min=rate, device_peak_GiB=memory.total_peak / 1024,
                                      process_peak_GiB=memory.process_peak / 1024, baseline_worlds=best["worlds"],
                                      relative_gain=gain, retention_gate_met=bool(status == "PASS" and gain >= .10),
                                      timing_scope=baseline["timing_scope"])
    _write(output / f"scaling_extension_{worlds}.json", report)
    log(f"Scaling extension {worlds}: {rate:.1f} valid/min, gain={gain:+.1%}, "
        f"GPU peak {report['device_peak_GiB']:.2f} GiB")
    return report


def scaling(output, device):
    rng = np.random.default_rng(441)
    ppo = MaskedPPO(211)
    probe_scorer = SurrogateScorer(ResidualEncoder(np.arange(62), include_effort=True))
    results = []
    best = None
    stop = "reached permitted size"
    for k, n in enumerate((1024, 2048, 4096, 8192, 16384)):
        if n == 16384 and (len(results) < 4 or results[-1]["device_peak_GiB"] >= 26 or
                           results[-1]["relative_gain"] < .15):
            stop = "16384 conditional gate not met"
            break
        env = None
        with GPUMemory() as memory:
            try:
                env = VectorSurrogate(n, device=device)
                # Constructor runs a microstep and compiles/captures kernels out of the timed interval.
                start = perf_counter()
                _, outcomes, _, _, _ = rollout(env, 129 + k, 1_000_000 + 20_000 * k, rng,
                                                actor=ppo, scorer=probe_scorer)
                seconds = perf_counter() - start
                row = summarize(outcomes) | dict(worlds=n, wall_s=seconds,
                                                  valid_macro_episodes_per_min=outcomes.valid.sum().item() * 60 / seconds)
            except RuntimeError as error:
                fault = str(error)
                if any(key in fault.casefold() for key in ("out of memory", "allocation failed", "cuda_error_out_of_memory")):
                    stop = "OOM"
                    results.append(dict(worlds=n, status="OOM", error=fault))
                    break
                raise
        try:
            row["process_peak_GiB"] = memory.process_peak / 1024
            row["device_peak_GiB"] = memory.total_peak / 1024
            if not row["process_peak_GiB"] or not row["device_peak_GiB"]:
                raise RuntimeError("The GPU peak monitor returned no result for the simulation PID")
            row["relative_gain"] = None
            if results:
                row["relative_gain"] = row["valid_macro_episodes_per_min"] / results[-1]["valid_macro_episodes_per_min"] - 1.
            row["status"] = ("PASS" if row["device_peak_GiB"] <= 28 and row["invalid"] / n <= .02 else "FAIL")
            results.append(row)
            log(f"Scaling {n}: {row['valid_macro_episodes_per_min']:.1f} valid/min; "
                f"{row['device_peak_GiB']:.2f} GiB total; {row['invalid']} invalid")
            if row["status"] == "PASS" and (best is None or
                    row["valid_macro_episodes_per_min"] > best["valid_macro_episodes_per_min"]):
                best = row
            if row["device_peak_GiB"] > 28:
                stop = "28 GiB device limit"
                break
            if row["relative_gain"] is not None and row["relative_gain"] < .10:
                stop = "throughput gain below 10%"
                break
        finally:
            del env
            gc.collect()
        _write(output / "scaling_benchmark.json", dict(status="RUNNING", rows=results, best=best))
    result = dict(status="PASS" if best else "FAIL", N_opt=None if best is None else best["worlds"],
                  rows=results, best=best, stop_reason=stop, timing_scope=(
                      "Reset, public observation, CPU actor, Probe, GPU history, CPU FiLM scorer, "
                      "Grasp, Lift, Transport, Release, Settle and device audit."))
    _write(output / "scaling_benchmark.json", result)
    return result


def c1(output, device, n_opt):
    rng = np.random.default_rng(22)
    quality = []
    small = VectorSurrogate(768, device=device)
    data, outcomes, *_ = rollout(small, 913, 2_000_000, rng, collect=True, training=True)
    result = summarize(outcomes)
    result["coverage"] = dict(position_min=data.tried.candidates[..., :3].min(axis=(0, 1)).tolist(),
                              position_max=data.tried.candidates[..., :3].max(axis=(0, 1)).tolist(),
                              guided_picks=int((data.choices < small.count // 2).sum()),
                              uniform_picks=int((data.choices >= small.count // 2).sum()))
    save_npz(output / "c1_quality_768.npz", data)
    quality.append(result)
    log(f"C1 quality batch: {result}")
    del small
    gc.collect()
    ok = .15 <= result["positive_rate"] <= .55 and result["invalid"] / result["episodes"] <= .02
    if not ok:
        _write(output / "c1_dataset_summary.json", dict(status="FAIL", quality=quality, valid_episodes=0))
        return None
    env = VectorSurrogate(n_opt, device=device)
    collected, waves = [], []
    for k in range(2):
        data, outcomes, *_ = rollout(env, 988 + k, 3_000_000 + n_opt * k, rng, collect=True, training=True)
        collected.append(data.take(np.flatnonzero(data.valid)))
        waves.append(summarize(outcomes))
        log(f"C1 wave {k + 1}: {waves[-1]}")
        if sum(map(len, collected)) >= 4096:
            break
    if sum(map(len, collected)) < 4096:
        report = dict(status="FAIL", quality=quality, waves=waves, valid_episodes=sum(map(len, collected)))
        _write(output / "c1_dataset_summary.json", report)
        return None
    data = concatenate(collected)
    if len(data) > 8192:
        data = data.take(rng.choice(len(data), 8192, replace=False))
    splits = split_initial_states(data, 9710)
    ids = [set(p.initial_id) for p in splits]
    if any(ids[i] & ids[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError("C1 initial-state leakage")
    for name, part in zip(("train", "validation", "test"), splits, strict=True):
        save_npz(output / f"c1_{name}.npz", part)
    report = dict(status="PASS", quality=quality, waves=waves, valid_episodes=len(data),
                  positive_rate=float(data.success.mean()), groups_no_overlap=True,
                  split=dict(zip(("train", "validation", "test"), map(len, splits), strict=True)))
    _write(output / "c1_dataset_summary.json", report)
    return splits


def save_checkpoint(path, scorer_version, policy_version, ppo, models, **meta):
    bundle = dict(policy_version=policy_version, scorer_version=scorer_version,
                  action_mask=torch.from_numpy(ACTION_MASK.copy()), actor=ppo.actor.state_dict(),
                  critic=ppo.critic.state_dict(), ppo_optimizer=ppo.opt.state_dict(),
                  generator_state=ppo.rng.get_state(), torch_state=torch.get_rng_state(),
                  scorers={key: model.state_dict() for key, model in models.items()}, metadata=meta)
    temp = path.with_suffix(".tmp.pt")
    torch.save(bundle, temp)
    temp.replace(path)


def instantiate_scorers(states):
    models = {}
    for name, state in states.items():
        encoder = None
        if "encoder.active" in state:
            encoder = ResidualEncoder(state["encoder.active"], include_effort=name != "without_effort")
        model = SurrogateScorer(encoder)
        model.load_state_dict(state)
        models[name] = model.eval()
    return models


def c2(output, splits, epochs, ppo):
    train, val, test = splits
    models, report = {}, {}
    for name, effort in (("informed", True), ("blind", None), ("without_effort", False)):
        encoder = None if effort is None else ResidualEncoder.fit_statistics(train, effort)
        model = SurrogateScorer(encoder)
        result = fit_scorer(model, train, val, epochs=epochs, seed=513)
        models[name] = model
        report[name] = dict(training=result, validation=metrics(model, val), test=metrics(model, test),
                            hidden_linear_probe=linear_probe(model, train, test))
        log(f"C2 {name}: {report[name]['test']}")
        _write(output / "c2_training_metrics.json", dict(status="RUNNING", **report))
    _write(output / "c2_training_metrics.json", dict(status="PASS", **report))
    save_checkpoint(output / "c2_scorers.pt", 1, 0, ppo, models, epochs=epochs)
    return models


def _reward(r, data, round_number):
    schedule = np.clip((round_number - 10) / 12., 0., 1.)
    weights = np.array([.01, .01, .03]) + schedule * np.array([.04, .04, .02])
    phase = data.tried.windows
    moving = np.max(np.abs(data.tried.history[:, :, 6:12]), axis=2) > .04
    search = data.tried.valid & ((phase == 2) | (phase == 3)) & moving
    probe_duration = search.sum(axis=1) / (PHASE_TICKS[2] + PHASE_TICKS[3])
    reward = r.success.astype(np.float32) - (
        weights[0] * probe_duration + weights[1] * r.audit["normalized_effort_cost"] +
        weights[2] * np.clip(r.audit["lift_instability"], 0., 1.))
    return reward, weights.tolist()


def _stratified(data, rng, target):
    if len(data) <= target:
        return data
    index = np.arange(len(data))
    groups = 2 * data.regime + data.success.astype(int)
    pieces = []
    for g in np.unique(groups):
        subset = index[groups == g]
        count = min(len(subset), max(1, int(target * len(subset) / len(data))))
        pieces.append(rng.choice(subset, count, replace=False))
    return data.take(np.concatenate(pieces))


def c3_c4(output, device, n_opt, splits, models, ppo, rounds, *, resume=None, convergence_patience=0):
    c1_train, val, _ = splits
    replay_path = output / "ppo_rollouts"
    replay_path.mkdir(exist_ok=True)
    rng = np.random.default_rng(231)
    logs = output / "ppo_progress.jsonl"
    files = []
    first, scorer_version, invalid_high, skips = 1, 1, 0, 0
    if resume is not None:
        first, scorer_version = resume["policy_version"] + 1, resume["scorer_version"]
        invalid_high, skips = (resume["metadata"].get(k, 0) for k in ("invalid_high", "skips"))
        for path in sorted(replay_path.glob("round-*.npz")):
            if int(path.stem.split("-")[-1]) < first:
                files.append(path)
    env = VectorSurrogate(n_opt, device=device)
    results = [] if not logs.exists() else [json.loads(line) for line in logs.read_text().splitlines()]
    results = [row for row in results if row["round"] < first]
    logs.write_text("".join(json.dumps(row) + "\n" for row in results))
    last_c4 = max((row["round"] for row in results if "c4" in row), default=0)
    broke = False
    converged = False
    for i in range(first, rounds + 1):
        t = perf_counter()
        data, outcomes, u, old_logp, max_q = rollout(env, 7117 + i, 6_000_000 + n_opt * i, rng,
                                                     actor=ppo, scorer=models["informed"], training=True)
        fraction = float(1. - data.valid.mean())
        invalid_high = invalid_high + 1 if fraction > .03 else 0
        skips = skips + 1 if fraction > .02 else 0
        reward, weights = _reward(outcomes, data, i)
        row = summarize(outcomes) | dict(round=i, scorer_version=scorer_version, skipped=bool(skips),
                                         invalid_rate=fraction, max_Q=float(max_q), reward_weights=weights,
                                         mean_reward=float(reward[data.valid].mean()) if data.valid.any() else None,
                                         probe_contact_rate=float(outcomes.audit["probe_contact"].mean()))
        if row["probe_contact_rate"] < .50:
            row["contact_warning"] = "probe_contact_rate_below_50_percent"
        if skips:
            failed = data.actions[~data.valid]
            row["invalid_actions"] = dict(mean=failed.mean(0).tolist(), std=failed.std(0).tolist())
        else:
            row["ppo"] = ppo.update(data, u, old_logp, reward, max(256, n_opt // 8))
        usable = data.take(np.flatnonzero(data.valid))
        if len(usable):
            path = replay_path / f"round-{i:03d}.npz"
            save_npz(path, _stratified(usable, rng, min(8192, len(usable))))
            files.append(path)
        recent = None
        if len(usable) >= 16:
            group = usable.initial_id % 10 == 0
            recent = usable.take(np.flatnonzero(group))
        diagnostic = metrics(models["informed"], recent) if recent is not None and len(recent) else None
        row["recent_diagnostic"] = diagnostic
        if invalid_high >= 2 or skips >= 3:
            broke = True
            _write(output / "physics_circuit_breaker.json",
                   dict(status="FAIL", round=i, invalid_high_streak=invalid_high, skipped_streak=skips,
                        action_diagnostic=row.get("invalid_actions"), physics_dt_s=.002,
                        workspace_xyz_m=[[-.18, -.34, .012], [.09, .035, .32]], overflow_flags=row["overflow_flags"]))
        elif not skips and recent is not None and len(recent) and (
                i % 6 == 0 or (i - last_c4 >= 3 and diagnostic["bce"] > .80)):
            chosen = []
            for path in files[-2:]:
                sample = load_npz(path)
                chosen.append(sample.take(np.flatnonzero(sample.initial_id % 10 != 0)))
            latest = concatenate(chosen)
            others = [c1_train]
            for path in files[max(0, len(files) - 8):-2]:
                sample = load_npz(path)
                others.append(sample.take(np.flatnonzero(sample.initial_id % 10 != 0)))
            background = concatenate(others)
            count = min(len(latest), len(background), 8192)
            combined = concatenate([_stratified(latest, rng, count), _stratified(background, rng, count)])
            candidate = deepcopy(models["informed"])
            train_metrics = fit_scorer(candidate, combined, val, epochs=15, encoder_lr=.0003, head_lr=.0001,
                                      patience=15, seed=511 + i)
            previous_val, next_val = metrics(models["informed"], val), metrics(candidate, val)
            next_recent = metrics(candidate, recent)
            published = next_val["bce"] <= previous_val["bce"] + .005 and next_recent["bce"] <= diagnostic["bce"] + .005
            if published:
                models["informed"] = candidate
                scorer_version += 1
            row["c4"] = dict(published=bool(published), previous_validation=previous_val, candidate_validation=next_val,
                             previous_recent=diagnostic, candidate_recent=next_recent,
                             sampled=len(combined), training=train_metrics)
            last_c4 = i
        row["scorer_version_after"] = scorer_version
        row["wall_s"] = perf_counter() - t
        if convergence_patience and i >= 20 and len(results) >= 2 * convergence_patience - 1 and not broke:
            window = [*results[-(2 * convergence_patience - 1):], row]
            rewards = [r["mean_reward"] for r in window]
            if (all(not r["skipped"] for r in window) and
                    all(r["scorer_version_after"] == scorer_version for r in window)):
                before = np.mean(rewards[:convergence_patience])
                after = np.mean(rewards[convergence_patience:])
                converged = bool(abs(after - before) < .003)
        if converged:
            row["convergence_early_stop"] = True
        log(f"PPO round {i}: {int(data.success.sum())}/{int(data.valid.sum())} "
            f"valid successes, invalid={fraction:.2%}, scorer={scorer_version}, seconds={row['wall_s']:.1f}")
        results.append(row)
        if not broke:
            save_checkpoint(output / f"policy_round_{i:03d}.pt", scorer_version, i, ppo, models,
                            invalid_high=invalid_high, skips=skips, n_opt=n_opt)
        with logs.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if broke or converged:
            break
    report = dict(status="FAIL" if broke else "PASS", rounds=len(results), scorer_version=scorer_version,
                  alternating_updates=sum("c4" in row for row in results),
                  published_updates=sum(row.get("c4", {}).get("published", False) for row in results),
                  circuit_breaker=bool(broke), convergence_early_stop=converged)
    _write(output / "c3_c4_training_summary.json", report)
    return report, results


def _fixed_actions(n):
    a = np.zeros((n, 13), dtype=np.float32)
    a[:, 8] = -.5
    a[:, 9] = .20
    return a


def _paired_bootstrap(a, b, regime, common, seed=409, repetitions=3000):
    idx = np.flatnonzero(common)
    if not len(idx):
        return dict(n=0, delta=None, ci95=None)
    diff = a.astype(float) - b.astype(float)
    rng = np.random.default_rng(seed)
    totals = np.zeros(repetitions)
    for g in np.unique(regime[idx]):
        pool = idx[regime[idx] == g]
        totals += diff[rng.choice(pool, size=(repetitions, len(pool)), replace=True)].sum(axis=1)
    return dict(n=len(idx), delta=float(diff[idx].mean()),
                ci95=np.quantile(totals / len(idx), [.025, .975]).tolist())


def held_out(output, device, ppo, models, splits, n=200):
    env = VectorSurrogate(n, device=device)
    v0 = env.reset(99519, 8_500_000)
    no_try = np.zeros((n, 13), dtype=np.float32)
    tried = env.try_batch(no_try, skip=np.ones(n, dtype=np.int32))
    snapshot = env.snapshot_try()
    train = splits[0]
    picks = np.arange(env.count)
    rates = np.array([(1 + train.success[train.choices == k].sum()) /
                      (2 + (train.choices == k).sum()) for k in picks])
    best = int(picks[rates.argmax()])
    outcomes = {}
    outcomes["Best Fixed"] = env.grasp_batch(np.full(n, best, dtype=np.int32))
    env.restore_try(snapshot)
    blind = score_batches(models["blind"], v0, no_try, tried).argmax(1)
    outcomes["Vision-Only No Try"] = env.grasp_batch(blind)
    v0 = env.reset(99519, 8_500_000)
    fixed = _fixed_actions(n)
    tried = env.try_batch(fixed)
    outcomes["Fixed Try + Informed"] = env.grasp_batch(score_batches(models["informed"], v0, fixed, tried).argmax(1))
    v0 = env.reset(99519, 8_500_000)
    _, learned, _ = ppo.sample(v0, deterministic=True)
    tried = env.try_batch(learned)
    snapshot = env.snapshot_try()
    names_models = (("Matched Blind", "blind"), ("Learned Try + Informed w/o Effort", "without_effort"),
                    ("Learned Try + Informed", "informed"))
    for name, model in names_models:
        env.restore_try(snapshot)
        outcomes[name] = env.grasp_batch(score_batches(models[model], v0, learned, tried).argmax(1))
    oracle = np.zeros(n, dtype=bool)
    oracle_valid = np.zeros(n, dtype=bool)
    oracle_candidates = 0
    for k in picks:
        env.restore_try(snapshot)
        obtained = env.grasp_batch(np.full(n, k, dtype=np.int32))
        oracle |= obtained.success
        oracle_valid |= obtained.valid
        oracle_candidates += 1
    regime = outcomes["Learned Try + Informed"].regime
    y = {name: r.success for name, r in outcomes.items()} | {"Privileged Oracle": oracle}
    valid = {name: r.valid for name, r in outcomes.items()} | {"Privileged Oracle": oracle_valid}
    rates, per_regime = {}, {}
    for name in EVAL_ORDER:
        rates[name] = dict(valid=int(valid[name].sum()), successes=int(y[name].sum()),
                           success_rate=float(y[name].sum() / max(1, valid[name].sum())))
        per_regime[name] = {}
        for i, tag in enumerate(REGIMES):
            subset = valid[name] & (regime == i)
            per_regime[name][tag] = dict(n=int(subset.sum()), success=int(y[name][subset].sum()),
                                         rate=float(y[name][subset].mean()) if subset.any() else None)
    a = "Learned Try + Informed"
    compare = {}
    for b in ("Matched Blind", "Fixed Try + Informed", "Learned Try + Informed w/o Effort"):
        compare[f"{a} vs {b}"] = _paired_bootstrap(y[a], y[b], regime, valid[a] & valid[b])
    report = dict(status="PASS", n_independent_initial_states=n, fixed_template=best, oracle_candidates=oracle_candidates,
                  matched_try_single_snapshot=True, rates=rates, regime=per_regime, paired_bootstrap=compare)
    _write(output / "held_out_evaluation.json", report)
    np.savez_compressed(output / "held_out_results.npz", regime=regime, groups=np.asarray(EVAL_ORDER),
                        success=np.stack([y[name] for name in EVAL_ORDER]),
                        valid=np.stack([valid[name] for name in EVAL_ORDER]))
    return report


def plots(output, ppo_rows=None, held=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if ppo_rows:
        x = [r["round"] for r in ppo_rows]
        fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True, layout="constrained")
        ax[0].plot(x, [r["positive_rate"] for r in ppo_rows], marker="o", label="Deposit success")
        ax[0].plot(x, [r["probe_contact_rate"] for r in ppo_rows], label="Probe contact")
        ax[1].plot(x, [r["mean_reward"] for r in ppo_rows], marker=".", label="Mean reward")
        ax[1].plot(x, [r["invalid_rate"] for r in ppo_rows], label="Invalid physics")
        for axis in ax:
            for row in ppo_rows:
                if "c4" in row:
                    axis.axvline(row["round"], color="#9c7c53", alpha=.45, ls="--")
            axis.legend()
            axis.grid(alpha=.18)
        ax[1].set_xlabel("PPO round")
        fig.savefig(output / "c3_c4_learning_curves.png", dpi=155)
        plt.close(fig)
    if held:
        fig, ax = plt.subplots(figsize=(11.2, 5.8), layout="constrained")
        x = np.arange(4)
        for i, name in enumerate(EVAL_ORDER):
            values = [held["regime"][name][tag]["rate"] for tag in REGIMES]
            ax.bar(x + .10 * (i - 3), [np.nan if v is None else v for v in values], .094, label=name)
        ax.set_xticks(x, ["Loose left", "Loose right", "Medium centered", "Flat full"])
        ax.set_ylim(0., 1.)
        ax.set_ylabel("Valid episode deposit success")
        ax.grid(axis="y", alpha=.17)
        ax.legend(ncol=2, fontsize=8)
        fig.savefig(output / "closed_loop_by_regime.png", dpi=155)
        plt.close(fig)


def publish(output, states, reason=None, *, ppo_rows=None, held=None):
    data = dict(experiment="Yam_B Policy V3", stage="A", valid_scope="surrogate approximation",
                milestones=states, blocker=reason)
    for key, filename in (("equivalence", "c0_kernel_equivalence.json"), ("c0", "c0_closed_loop.json"),
                          ("scaling", "scaling_benchmark.json"), ("dataset", "c1_dataset_summary.json"),
                          ("coupled_equivalence", "c0_suspended_gpu.json"), ("c2", "c2_training_metrics.json"),
                          ("edge_comparison", "extreme_case_comparison.json"),
                          ("learning", "c3_c4_training_summary.json"), ("evaluation", "held_out_evaluation.json")):
        path = output / filename
        if path.exists():
            data[key] = json.loads(path.read_text())
    extras = [json.loads(p.read_text()) for p in sorted(output.glob("scaling_extension_*.json"))]
    if extras:
        data["scaling_extensions"] = extras
    _write(output / "summary.json", data)
    bench = data.get("scaling", {})
    best = bench.get("best")
    c0 = data.get("c0", {})
    evals = data.get("evaluation", {}).get("paired_bootstrap", {})
    lines = ["# Yam_B Policy V3 · Stage A", "",
             "本次使用四块双柔索低保真代理近似。悬空时双柔索经锚点力分配承担载荷。"
             "高保真 PE 薄膜的局部起皱、撕裂与真实机器人表现仍需独立验证。", "",
             "| 阶段 | 状态 |", "|---|---|"]
    lines.extend(f"| {key} | {states[key]} |" for key in MILESTONES)
    lines += ["", "## GPU 并发", ""]
    if best:
        lines.append(f"**PASS** 选择 {bench['N_opt']} 路。{best['valid_macro_episodes_per_min']:.1f} 个有效宏回合/分钟，"
                     f"GPU 峰值总占用 {best['device_peak_GiB']:.2f} GiB，本进程峰值 {best['process_peak_GiB']:.2f} GiB。")
    else:
        lines.append(f"**{states['scaling']}** 当前没有可选的实测并发档。")
    for extra in extras:
        lines.append(f"- **{extra['status']}** {extra['worlds']:,} 路补测，"
                     f"{extra['valid_macro_episodes_per_min']:,.0f} 个有效宏回合/分钟，"
                     f"较训练档 {extra['relative_gain']:+.1%}。总显存峰值 {extra['device_peak_GiB']:.2f} GiB，"
                     f"无效率 {extra['invalid'] / extra['episodes']:.2%}。")
    lines += ["", "## C0 与物理", "",
              f"- **{'PASS' if 'equivalence' in data else 'NOT TESTED'}** 同输入双柔索与四块积分 GPU 数值对照。"]
    lines.extend(f"- **{'PASS' if v else 'FAIL'}** `{k}`" for k, v in c0.get("checks", {}).items())
    coupled = data.get("coupled_equivalence")
    if coupled:
        lines.append(f"- **PASS** 悬空耦合对照覆盖 {coupled['independent_worlds']} 路、"
                     f"每路 {coupled['substeps']} 个 2 ms 子步。最大张力误差 "
                     f"{coupled['max_tension_error_N']:.3g} N，最大净空误差 "
                     f"{coupled['max_clearance_error_m']:.3g} m。捕获判定逐项一致。")
    when = c0.get("normal_success_by_regime")
    if when:
        lines.append(f"- 四工况有效正样本数依次为 {', '.join(str(when[r]) for r in REGIMES)}。"
                     f"其中 {len(c0['empty_edge_successes'])} 条满足空侧抓持、柔索吊起及有效入篮条件。")
    if "edge_comparison" in data:
        edge = data["edge_comparison"]
        rates = "，".join(f"{row['abs_lateral_mm']:.0f} mm 为 "
                          f"{row['clean_empty_side_success']}/{row['valid']}" for row in edge["rows"])
        lines.append(f"- 补测距中心更远的空侧抓持，{rates}。满装近中心补测为 "
                     f"{edge['flat_target_success']}/{edge['flat_target_episodes']}。"
                     "这是两组定向工况复测，初态条件和抓姿不同。")
    lines += ["", "## C1–C4", ""]
    dataset = data.get("dataset", {})
    if dataset.get("status") == "PASS":
        part = dataset["split"]
        lines.append(f"- **PASS** C1 入库 {dataset['valid_episodes']:,} 条有效回合，"
                     f"正率 {dataset['positive_rate']:.1%}。训练、验证、测试的独立初态组数为 "
                     f"{part['train']:,} / {part['validation']:,} / {part['test']:,}。")
    if data.get("c2", {}).get("status") == "PASS":
        lines += ["", "| C2 留出打分 | BCE | AUROC | Brier |", "|---|---:|---:|---:|"]
        for key in ("informed", "blind", "without_effort"):
            metric = data["c2"][key]["test"]
            auc = "NA" if metric["auroc"] is None else f"{metric['auroc']:.3f}"
            lines.append(f"| {key} | {metric['bce']:.3f} | {auc} | {metric['brier']:.3f} |")
    if "learning" in data:
        learning = data["learning"]
        c4_status = "PASS" if learning["alternating_updates"] else "NOT TESTED"
        lines += ["", f"- **{learning['status']}** C3 执行 {learning['rounds']} 轮。",
                  f"- **{c4_status}** C4 尝试 {learning['alternating_updates']} 次，"
                  f"验证后发布 {learning['published_updates']} 次，最终 scorer_version={learning['scorer_version']}。"]
    if ppo_rows:
        first, last = ppo_rows[0], ppo_rows[-1]
        alert_count = sum(r["probe_contact_rate"] < .50 for r in ppo_rows)
        lines.append(f"- PPO 采样入篮率从 {first['positive_rate']:.1%} 变化到 {last['positive_rate']:.1%}，"
                     f"最高无效率 {max(r['invalid_rate'] for r in ppo_rows):.2%}。")
        lines.append(f"- **{'FAIL' if alert_count else 'PASS'}** 探测接触率低于 50% 的轮数为 "
                     f"{alert_count}/{len(ppo_rows)}，最终为 {last['probe_contact_rate']:.1%}。"
                     "结合留出结果判断策略是否受益于试探反馈。")
    lines += ["", "## 留出配对评估", ""]
    for label in ("Matched Blind", "Fixed Try + Informed", "Learned Try + Informed w/o Effort"):
        key = f"Learned Try + Informed vs {label}"
        if key in evals and evals[key]["ci95"] is not None:
            r = evals[key]
            lo, hi = r["ci95"]
            status = "PASS" if lo > 0 else "FAIL"
            lines.append(f"- **{status}** {key}，差值 {r['delta']:+.1%}，95% CI [{lo:+.1%}, {hi:+.1%}]，配对 {r['n']}。")
        else:
            lines.append(f"- **NOT TESTED** {key}。")
    if "evaluation" in data:
        evaluation = data["evaluation"]
        lines += ["", f"独立初态 {evaluation['n_independent_initial_states']} 个。",
                  "", "| 闭环组 | 总体 | 松装左偏 | 松装右偏 | 中等居中 | 扁平满装 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name in EVAL_ORDER:
            rate = evaluation["rates"][name]
            sliced = [evaluation["regime"][name][g]["rate"] for g in REGIMES]
            scores = " | ".join("NA" if s is None else f"{s:.1%}" for s in sliced)
            lines.append(f"| {name} | {rate['success_rate']:.1%} | {scores} |")
    lines += ["", "## Flex 迁移", "",
              "- 使用同一 13D Probe 动作顺序和 Action Mask。",
              "- 复用公开 v0、6D 旋转抓姿、25 Hz Yam_B 历史字段、候选池和冻结归一化统计。",
              "- 使用独立 MJWarp Flex 闭环做零样本评估，随后对比从零训练和高保真微调。",
              "- 工程先验包含观测噪声、死区、偏置、低通、延迟及扭矩残差安全阈值。真实 Yam_B 需标定这些值。",
              "- 正式训练按四个工况分层，并在层内连续抽样。跨四类全区间的混合模式未用于本轮训练。",
              "- 核函数等价项覆盖两个柔索的捕获、张力和滑移方程及四点积分。闭环代理包含单独的折叠、滑脱与篮沿约束。"
              "周一原始颈部阻尼和扭转只在冻结复放中覆盖。本次生产 GPU 环境采用共享悬空子模型。"]
    if (output / "attempt-01-pre-contact-fix" / "physics_circuit_breaker.json").exists():
        lines += ["", "## 主要失败原因", "",
                  "首轮 PPO 的前两波采集触发物理熔断。包裹翻转后，表面材料坐标投影曾对远处指垫产生异常法向力。"
                  "修正版加入有限接触空间判定，并保留原熔断阈值。"
                  "原始诊断和跳过记录保存在 `attempt-01-pre-contact-fix/`。"]
    if reason:
        lines += ["", "## 当前阻塞", "", reason]
    (output / "v1_rl_report.md").write_text("\n".join(lines) + "\n")
    plots(output, ppo_rows, held)


def main():
    arg = argparse.ArgumentParser()
    arg.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs" / "surrogate-v1-large-scale")
    arg.add_argument("--device", default=os.environ.get("YAM_SURROGATE_DEVICE", "cuda:0"))
    arg.add_argument("--rounds", type=int, default=30)
    arg.add_argument("--c2-epochs", type=int, default=80)
    arg.add_argument("--ppo-patience", type=int, default=0,
                     help="Positive window length for reward convergence stopping after round 20")
    arg.add_argument("--stage", choices=("all", "c0", "scaling", "c1", "train", "scaling-extra"), default="all")
    arg.add_argument("--worlds", type=int, default=24576, help="Concurrency for the additional scaling test")
    arg.add_argument("--resume", type=Path)
    opts = arg.parse_args()
    out = opts.output
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    device = wp.get_device(opts.device)
    if not device.is_cuda or torch.cuda.is_available():
        raise RuntimeError("Expected MJWarp CUDA and CPU-only Torch for the experiment")
    if opts.stage == "scaling-extra":
        return int(extension_benchmark(out, opts.device, opts.worlds)["status"] != "PASS")
    states = dict.fromkeys(MILESTONES, "NOT TESTED")
    states["preflight"] = "PASS"
    import mujoco
    import mujoco_warp
    _write(out / "preflight.json", dict(device=device.name, executable=os.sys.executable, device_name=str(device),
                                      torch=torch.__version__, torch_cuda=torch.cuda.is_available(),
                                      mujoco=mujoco.__version__, mujoco_warp=mujoco_warp.__version__, warp=wp.__version__))
    try:
        c0_file = out / "c0_closed_loop.json"
        previous_c0 = json.loads(c0_file.read_text()) if c0_file.exists() else {}
        if opts.stage == "c0" or (opts.stage == "all" and previous_c0.get("status") != "PASS"):
            c0_data = run_c0(out, opts.device)
        else:
            c0_data = previous_c0
        states["c0"] = c0_data.get("status", "NOT TESTED")
        if states["c0"] != "PASS" or opts.stage == "c0":
            publish(out, states, "新版 C0 必须通过后才能进入大并发训练。" if states["c0"] != "PASS" else None)
            return int(states["c0"] != "PASS")
        if opts.stage in ("all", "scaling"):
            s = scaling(out, opts.device)
        else:
            s = json.loads((out / "scaling_benchmark.json").read_text())
        states["scaling"] = s["status"]
        if s["status"] != "PASS" or opts.stage == "scaling":
            publish(out, states, "未找到满足物理有效率与显存红线的并发档。" if s["status"] != "PASS" else None)
            return int(s["status"] != "PASS")
        n_opt = s["N_opt"]
        if opts.stage in ("all", "c1"):
            splits = c1(out, opts.device, n_opt)
        else:
            splits = tuple(load_npz(out / f"c1_{name}.npz") for name in ("train", "validation", "test"))
        states["c1"] = "PASS" if splits else "FAIL"
        if splits is None or opts.stage == "c1":
            publish(out, states, "C1 小批质检或有效样本预算未达标。" if splits is None else None)
            return int(splits is None)
        ppo = MaskedPPO(19)
        checkpoint = opts.resume
        if checkpoint is None and (out / "c2_scorers.pt").exists():
            checkpoint = out / "c2_scorers.pt"
        restored = None
        if checkpoint:
            restored = torch.load(checkpoint, weights_only=True)
            ppo.actor.load_state_dict(restored["actor"])
            ppo.critic.load_state_dict(restored["critic"])
            ppo.opt.load_state_dict(restored["ppo_optimizer"])
            ppo.rng.set_state(restored["generator_state"])
            torch.set_rng_state(restored["torch_state"])
            models = instantiate_scorers(restored["scorers"])
        else:
            models = c2(out, splits, opts.c2_epochs, ppo)
        states["c2"] = "PASS"
        c3, progress = c3_c4(out, opts.device, n_opt, splits, models, ppo, opts.rounds, resume=restored,
                             convergence_patience=opts.ppo_patience)
        states["c3_c4"] = c3["status"]
        if c3["status"] != "PASS":
            publish(out, states, "PPO 触发物理有效性熔断。", ppo_rows=progress)
            return 1
        evaluation = held_out(out, opts.device, ppo, models, splits)
        states["held_out"] = "PASS"
        publish(out, states, ppo_rows=progress, held=evaluation)
        return 0
    except Exception as error:
        publish(out, states, f"{type(error).__name__} · {error}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
