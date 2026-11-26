"""C0 validation and offline single-world rendered replay."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .surrogate_env import VectorSurrogate


REGIMES = ("loose_left", "loose_right", "medium_centered", "flat_full")


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _fixed_try(n, *, deep=False):
    a = np.zeros((n, 13), dtype=np.float32)
    a[:, 8] = -1. if deep else -.5
    return a


def _scripted_grasps(v0, *, edge):
    n = len(v0)
    r = np.zeros((n, 16, 11), dtype=np.float32)
    r[:, :, 0] = np.linspace(-.026, .026, 16)
    r[:, :, 1] = np.linspace(-.028, .028, 16)
    if edge:
        r[:, :, 0] = np.where(np.arange(n)[:, None] % 4 == 0, 1., -1.) * (
            np.linspace(.052, .077, 16)[None])
        r[:, :, 1] = np.linspace(-.022, .012, 16)
    r[:, :, 2] = (.023 if edge else .020) - v0[:, 2, None]
    fx, fy = v0[:, 3:6], v0[:, 6:9]
    r[:, :, 3] = -fx[:, None, 1]
    r[:, :, 4] = -fy[:, None, 1]
    r[:, :, 6] = -fx[:, None, 0]
    r[:, :, 7] = -fy[:, None, 0]
    r[:, :, 9] = .074
    r[:, :, 10] = .69
    return r


def summarize(r):
    valid = r.valid
    return dict(episodes=len(valid), valid=int(valid.sum()), invalid=int((~valid).sum()),
                success=int(r.success.sum()), positive_rate=float(r.success.sum() / max(valid.sum(), 1)),
                retracted=int(np.count_nonzero(r.audit["retracted"])),
                probe_contacts=int(np.count_nonzero(r.audit["probe_contact"])),
                probe_cutoffs=int(np.count_nonzero(r.audit["probe_unloaded"])),
                direct_force_nonzero=int(np.count_nonzero(r.audit["robot_payload_direct_force"])),
                peak_direct_force_N=float(r.audit["robot_payload_direct_force"].max()),
                max_clearance_m=float(r.audit["peak_clearance_m"].max()),
                invalid_reasons={str(k): int(np.count_nonzero(r.audit["invalid_reason"] == k))
                                 for k in np.unique(r.audit["invalid_reason"][~valid])},
                overflow_flags={str(k): int(np.count_nonzero(r.audit["overflow_flags"] == k))
                                for k in np.unique(r.audit["overflow_flags"][~valid])},
                success_by_regime=dict(zip(REGIMES, (int(r.success[r.regime == i].sum()) for i in range(4)))))


def _history_diagnostic(try_result, regime):
    use = (try_result.windows == 2) | (try_result.windows == 3)
    history = try_result.history[:, :, 30:36]
    x = np.concatenate((np.max(abs(history) * use[:, :, None], axis=1),
                        np.sqrt(np.sum(history**2 * use[:, :, None], axis=1) / use.sum(1)[:, None])), axis=1)
    first = np.arange(len(x)) % 8 < 4
    mean, std = x[first].mean(0), x[first].std(0)
    normalized = (x - mean) / np.maximum(std, .01)
    centers = np.stack([normalized[first & (regime == k)].mean(0) for k in range(4)])
    pred = ((normalized[~first, None] - centers[None])**2).sum(axis=2).argmin(1)
    return dict(held_out_nearest_centroid_accuracy=float((pred == regime[~first]).mean()),
                test_rows=len(pred), chance=.25, residual_rms_by_regime={
                    name: np.sqrt((x[regime == k, 6:]**2).mean(0)).tolist() for k, name in enumerate(REGIMES)})


def _drop_examples(trace, r, fault):
    lift = trace[:, :, 27] == 8
    active = trace[:, :, 29] > 0
    risen = (lift & active & (trace[:, :, 28] > .022)).any(1)
    active_until = np.where(lift & active, np.arange(trace.shape[1])[None], -1).max(1)
    falling = (lift & (~active) & (trace[:, :, 28] < .006) &
               (np.arange(trace.shape[1])[None] > active_until[:, None])).any(1)
    physical = r.valid & risen & falling & (r.audit["slipped_tethers"] > 0) & (r.audit["probe_cutoff"] != 2)
    return {name: np.flatnonzero(physical & (fault == mode)).tolist()
            for name, mode in (("insufficient_clamp", 1), ("eccentric_overload", 2))}


def render_demo(model, trajectory, frame_public, output):
    import imageio_ffmpeg
    import mujoco

    renderer = mujoco.Renderer(model, width=640, height=360)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [-.08, -.155, .065]
    camera.distance = .96
    camera.azimuth = 154.
    camera.elevation = -30.
    data = mujoco.MjData(model)
    codec = imageio_ffmpeg.write_frames(str(output), (640, 360), fps=25, codec="libx264",
                                      pix_fmt_in="rgb24", pix_fmt_out="yuv420p", macro_block_size=2)
    identity = np.eye(3).reshape(-1)
    codec.send(None)
    try:
        for frame in trajectory:
            data.qpos[:] = frame[:model.nq]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera)
            scene = renderer.scene
            centers = frame[8:20].reshape(4, 3)
            for i, pos in enumerate(centers):
                mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_BOX,
                                   np.array([.020, .016, max(.002, frame[20 + i])]), pos, identity,
                                   np.array([.84, .49 + .06 * i, .22, 1.]))
                scene.ngeom += 1
            if frame[29] > 0:
                pads = frame[24:27] - [0, 0, .009]
                for side in range(2):
                    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE,
                                       np.zeros(3), np.zeros(3), identity, np.array([.18, .78, .90, 1.]))
                    mujoco.mjv_connector(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_CAPSULE, .0017,
                                         pads + [.006 * (2 * side - 1), 0, 0],
                                         .5 * (centers[side] + centers[side + 2]))
                    scene.ngeom += 1
            codec.send(renderer.render())
    finally:
        codec.close()
        renderer.close()


def run_c0(output, device):
    from .surrogate_equivalence import evaluate

    start = perf_counter()
    equivalence = evaluate(device)
    _write(output / "c0_kernel_equivalence.json", equivalence)
    successes = np.zeros(4, dtype=np.int32)
    edge_samples = []
    faults = {}
    recorded = None
    entries = []
    env = VectorSurrogate(64, device=device, trace=True)
    for wave, variant in enumerate(("geometric", "edge", "deep", "fault")):
        v0 = env.reset(31 + 11 * wave, 40 + 128 * wave)
        action = _fixed_try(env.worlds, deep=variant == "deep")
        chosen = (np.arange(env.worlds) // 4) % 16
        override = None if variant in ("geometric", "deep") else _scripted_grasps(v0, edge=variant == "edge")
        tried = env.try_batch(action, override_candidates=override)
        if variant == "fault":
            modes = np.where(np.arange(env.worlds) % 8 < 4, 1, 2).astype(np.int32)
            env.set_c0_faults(modes)
        r = env.grasp_batch(chosen)
        trace = env.trace.numpy()
        rows = summarize(r)
        rows["wave"] = variant
        rows["telemetry_separation"] = _history_diagnostic(tried, r.regime)
        if variant == "fault":
            faults = _drop_examples(trace, r, modes)
        else:
            for mode in range(4):
                successes[mode] += int(r.success[r.regime == mode].sum())
        lift = (trace[:, :, 27] == 8) & (trace[:, :, 29] > 0) & (trace[:, :, 28] > .08)
        ratio = trace[:, :, 32] / (r.private[:, 0, None] * .4 * 9.81)
        rows["near_static_rope_weight_ratio"] = (
            float(np.median(ratio[lift])) if lift.any() else None)
        lateral = tried.candidates[np.arange(len(chosen)), chosen, 0]
        close = trace[:, :, 27] == 7
        empty_edge = (r.regime < 2) & (lateral * r.private[:, 1] < 0) & (abs(lateral) >= .05)
        clean_capture = np.max(np.where(close, trace[:, :, 30], 0.), axis=1) < .10
        for i in np.flatnonzero(r.success & empty_edge & clean_capture):
            edge_samples.append(dict(wave=variant, world=int(i), regime=REGIMES[r.regime[i]],
                                     candidate_lateral_m=float(lateral[i]),
                                     peak_clearance_m=float(r.audit["peak_clearance_m"][i])))
            if recorded is None:
                recorded = trace[i].copy(), v0[i].copy(), True
        if recorded is None and r.success.any():
            idx = int(np.flatnonzero(r.success)[0])
            recorded = trace[idx].copy(), v0[idx].copy(), False
        if recorded is not None and not recorded[2] and edge_samples and variant == edge_samples[-1]["wave"]:
            idx = edge_samples[-1]["world"]
            recorded = trace[idx].copy(), v0[idx].copy(), True
        entries.append(rows)
        np.savez_compressed(output / f"c0_{variant}_trace.npz", trace=trace, regime=r.regime, selected=chosen,
                            v0=v0, candidates=tried.candidates, **r.audit)
        print(f"C0 {variant} {rows}", flush=True)
    checks = dict(
        all_regimes_positive=bool((successes > 0).all()),
        at_least_ten_positive=bool(successes.sum() >= 10),
        empty_edge_lift_and_deposit=bool(edge_samples),
        insufficient_clamp_dropped=bool(faults.get("insufficient_clamp")),
        eccentric_overload_dropped=bool(faults.get("eccentric_overload")),
        measured_direct_force=any(row["direct_force_nonzero"] for row in entries),
        cutoff_and_retract=any(row["probe_cutoffs"] and row["retracted"] >= .95 * env.worlds for row in entries),
        histories_separable=any(row["telemetry_separation"]["held_out_nearest_centroid_accuracy"] >= .375
                                for row in entries))
    report = dict(status="PASS" if all(checks.values()) else "FAIL", checks=checks, waves=entries,
                  normal_success_by_regime=dict(zip(REGIMES, map(int, successes))), empty_edge_successes=edge_samples,
                  fault_drop_examples=faults, elapsed_s=perf_counter() - start)
    _write(output / "c0_closed_loop.json", report)
    if recorded is not None:
        np.savez_compressed(output / "c0_demo_trajectory.npz", trace=recorded[0], public_geometry=recorded[1],
                            empty_edge_trajectory=recorded[2])
        render_demo(env.model, recorded[0], recorded[1], output / "yam_surrogate_demo.mp4")
    return report


if __name__ == "__main__":
    import os

    out = Path("runs/surrogate-v1-large-scale")
    out.mkdir(parents=True, exist_ok=True)
    result = run_c0(out, os.environ.get("YAM_SURROGATE_DEVICE", "cuda:0"))
    print(json.dumps(result["checks"], indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
