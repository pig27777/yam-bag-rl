"""C0 numerical comparisons at the CPU reference / Warp device boundary."""

from dataclasses import replace

import numpy as np
import torch
import warp as wp

from ..bag.lumped_contact_proxy import ContactInputs, LumpedContactProxy, ProxyParams
from ..bag.macro_parcel_sim import MacroParcelSim, ParcelAction
from ..bag.replay_config import MONDAY, PAYLOAD
from ..bag.unified_payload import PayloadParamsX, PayloadState, UnifiedPayloadTemplate
from . import surrogate_kernels as gpu


def rope_parameters(values):
    p = gpu.RopeParameters()
    (p.budget, p.max_slack, p.gather_fraction, p.stiffness, p.damping, friction,
     p.slip_coefficient, p.capture_slack, p.gather_normal, p.capture_normal, p.release_normal,
     p.capture_depth, p.capture_alignment, p.gather_delay, p.gather_speed, rest) = (
        values[k] for k in ("S_bag", "s_max", "alpha", "k", "c_d", "mu_eff", "beta", "s_crit",
                            "F_min", "F_lock", "F_release", "h_min", "cos_theta_max",
                            "tau_gate", "v_gather_max", "L0"))
    p.friction = wp.vec2(*friction)
    p.rest = wp.vec2(*rest)
    return p


def payload_parameters(model, row):
    p = gpu.PayloadParameters()
    cfg = model.params
    p.half = wp.vec3(*cfg.bag_half_extents[row])
    p.height = wp.vec4(float(model.half_height[row, 0]))
    p.mass = wp.vec4(*model.mass[row])
    p.inv_mass = wp.vec4(*model.inv_mass[row])
    p.rest = gpu.Mat43(*model.rest[row].flatten())
    p.slide = float(model.slide[row, 0])
    p.stretch_alpha = float(model.stretch_alpha[row, 0])
    p.bend_alpha = float(model.bend_alpha[row, 0])
    p.bend_limit = float(model.bend_limit[row, 0])
    p.damping = float(cfg.internal_damping[row, 0])
    p.friction = float(cfg.table_friction[row, 0])
    p.rebound = float(model.rebound[row, 0])
    p.max_depth = wp.vec4(float(model.depth_max[row, 0]))
    p.soft_k = float(cfg.k_soft[row, 0])
    p.compression = float(cfg.c_comp[row, 0])
    p.hardening = float(cfg.hardening_gamma[row, 0])
    p.lateral_k = float(cfg.lateral_crush_k[row, 0])
    return p


def _compare_ropes(device, n, rng):
    values = dict(MONDAY["proxy"])
    params = ProxyParams.from_scalars(values)
    params = replace(params, **{k: v.repeat(n, 1) for k, v in vars(params).items()})
    ref = LumpedContactProxy(params, gpu.DT)
    state = ref.initial_state()
    state[:, :, 0] = torch.from_numpy(rng.uniform(.001, .019, (n, 2)))
    state[:, :, 1] = torch.from_numpy(rng.integers(0, 2, (n, 2)))
    state[:, :, 2:6] = torch.from_numpy(rng.uniform(0., .002, (n, 2, 4)))
    state[:, :, 3] += .016
    state_length = torch.full((n, 2), .065)
    # Include closed but slack, unloading, exhausted reserve, and new capture.
    inputs = []
    states = []
    wp_params = wp.array([rope_parameters(values)] * n, dtype=gpu.RopeParameters, device=device)
    anchor = torch.tensor(rng.uniform(-.02, .02, (n, 2, 3)), dtype=torch.float32)
    distance = torch.tensor(rng.uniform(.030, .16, (n, 2)), dtype=torch.float32)
    tip = anchor.clone()
    tip[:, :, 2] += distance
    speed = torch.tensor(rng.uniform(-.7, .7, (n, 2, 3)), dtype=torch.float32)
    anchor_speed = torch.tensor(rng.uniform(-.3, .3, (n, 2, 3)), dtype=torch.float32)
    normal = torch.tensor(rng.uniform(0., 12., (n, 2)), dtype=torch.float32)
    inward = torch.tensor(rng.uniform(-.03, .10, (n, 2)), dtype=torch.float32)
    feed = torch.tensor(rng.uniform(0., .02, (n, 2)), dtype=torch.float32)
    alignment = torch.tensor(rng.uniform(.6, 1., (n, 2)), dtype=torch.float32)
    mobility = torch.tensor(rng.uniform(1., 20., (n, 2)), dtype=torch.float32)
    budget = torch.tensor(rng.uniform(.002, .034, (n, 1)), dtype=torch.float32)
    for i in range(n):
        x = gpu.RopeInput()
        x.tip = gpu.Mat23(*tip[i].flatten())
        x.anchor = gpu.Mat23(*anchor[i].flatten())
        x.tip_velocity = gpu.Mat23(*speed[i].flatten())
        x.anchor_velocity = gpu.Mat23(*anchor_speed[i].flatten())
        x.normal = wp.vec2(*normal[i])
        x.inward = wp.vec2(*inward[i])
        x.feed = wp.vec2(*feed[i])
        x.alignment = wp.vec2(*alignment[i])
        x.mobility = wp.vec2(*mobility[i])
        x.budget = float(budget[i, 0])
        inputs.append(x)
        st = gpu.RopeState()
        st.contacts = gpu.Mat26(*state[i].flatten())
        st.reference = wp.vec2(*state_length[i])
        states.append(st)
    expected, reference = ref.step_with_reference(state, ContactInputs(
        tip, anchor, speed, anchor_speed, torch.zeros((n, 3)), normal, inward, feed, alignment, budget),
        state_length, mobility)
    x = wp.array(inputs, dtype=gpu.RopeInput, device=device)
    s = wp.array(states, dtype=gpu.RopeState, device=device)
    out = wp.empty(n, dtype=gpu.RopeResult, device=device)
    wp.launch(gpu.advance_ropes, n, [wp_params, x, s, False, out], device=device)
    result = out.numpy()
    errors = {}
    for name, actual, want in (
        ("tension_N", result["tension"], expected.tension),
        ("slip_rate_m_s", result["slip_rate"], expected.slip_rate),
        ("reference_m", result["state"]["reference"], reference),
        ("contact_state", result["state"]["contacts"], expected.state),
        ("rope_force_N", result["force"], expected.force),
    ):
        np.testing.assert_allclose(actual, want.numpy(), rtol=4e-6, atol=3e-6, err_msg=name)
        errors[name] = float(np.max(np.abs(actual - want.numpy())))
    np.testing.assert_array_equal(result["state"]["contacts"][:, :, 1], expected.state[:, :, 1])
    return errors


def _compare_blocks(device, n, rng):
    cfg = PayloadParamsX.from_presets(PAYLOAD, ["folded_clothes"] * n)
    cfg = replace(cfg, bag_half_extents=torch.tensor(rng.uniform(
        [.06, .07, .006], [.12, .14, .025], (n, 3)), dtype=torch.float32),
                  total_mass=torch.tensor(rng.uniform(.14, .45, (n, 1)), dtype=torch.float32))
    model = UnifiedPayloadTemplate(cfg)
    base = model.initial_state()
    offset = torch.tensor(rng.uniform(0., .06, (n, 1, 1)), dtype=torch.float32)
    pos = base.pos.clone()
    pos[:, :, 2:] += offset
    v = torch.tensor(rng.uniform(-.3, .3, (n, 4, 3)), dtype=torch.float32)
    initial = PayloadState(pos, v, base.d_surf)
    depths = model.depth_max * torch.tensor(rng.uniform(0., .75, (n, 4)), dtype=torch.float32)
    loads = torch.tensor(rng.uniform(-1.5, 2., (n, 4, 3)), dtype=torch.float32)
    states = []
    for i in range(n):
        s = gpu.PayloadState()
        s.position = gpu.Mat43(*pos[i].flatten())
        s.velocity = gpu.Mat43(*v[i].flatten())
        s.depth = wp.vec4(*base.d_surf[i])
        states.append(s)
    p = wp.array([payload_parameters(model, i) for i in range(n)], dtype=gpu.PayloadParameters, device=device)
    s = wp.array(states, dtype=gpu.PayloadState, device=device)
    f = wp.array(loads.numpy(), dtype=gpu.Mat43, device=device)
    d = wp.array(depths.numpy(), dtype=wp.vec4, device=device)
    out = wp.empty(n, dtype=gpu.PayloadResult, device=device)
    for _ in range(40):
        wp.launch(gpu.advance_blocks, n, [p, f, d, s, out], device=device)
        expected = model.step(initial, loads, depths)
        initial = expected.state
    result = out.numpy()
    errors = {}
    for name, actual, want, atol in (
        ("block_position_m", result["state"]["position"], expected.state.pos, 2e-5),
        ("clearance_m", result["clearance"], expected.clearance, 2e-5),
        ("block_velocity_m_s", result["state"]["velocity"], expected.state.vel, 4e-4),
        ("floor_support_N", result["support"], expected.support_N, 2e-3),
    ):
        np.testing.assert_allclose(actual, want.numpy(), rtol=2e-3, atol=atol, err_msg=name)
        errors[name] = float(np.max(np.abs(actual - want.numpy())))
    return errors


def _strict_release(device):
    n = 5
    params = rope_parameters(MONDAY["proxy"])
    loaded = gpu.RopeState()
    loaded.reference = params.rest
    loaded.contacts = gpu.Mat26(.010, 1., 0., .1, .01, 0., .010, 1., 0., .1, .01, 0.)
    inputs = []
    for i in range(n):
        x = gpu.RopeInput()
        # Frozen reference is exactly representable for the equality test.
        height = params.rest[0] if i == 0 else (.04 if i == 1 else .08)
        x.tip = gpu.Mat23(0., 0., height, .01, 0., height)
        x.anchor = gpu.Mat23(0., 0., 0., .01, 0., 0.)
        speed = (4., 0., .7) if i == 4 else (0., 0., .1)
        x.tip_velocity = gpu.Mat23(*speed, *speed)
        x.normal = wp.vec2(.15 if i == 3 else 6.)
        x.inward = wp.vec2(0.)
        x.feed = wp.vec2(.01)
        x.alignment = wp.vec2(1.)
        x.mobility = wp.vec2(1.)
        x.budget = params.budget
        x.opening = i == 2
        x.closing = False
        x.slip_limit = 1.
        inputs.append(x)
    result = wp.empty(n, dtype=gpu.RopeResult, device=device)
    wp.launch(gpu.advance_ropes, n, [
        wp.array([params] * n, dtype=gpu.RopeParameters, device=device),
        wp.array(inputs, dtype=gpu.RopeInput, device=device),
        wp.array([loaded] * n, dtype=gpu.RopeState, device=device), True, result,
    ], device=device)
    data = result.numpy()
    np.testing.assert_array_equal(data["tension"], 0.)
    np.testing.assert_array_equal(data["state"]["contacts"][:, :, 1],
                                  [[1., 1.], [1., 1.], [0., 0.], [0., 0.], [0., 0.]])
    return dict(zero_at_rest=True, zero_when_slack=True, opening_release=True,
                insufficient_clamp_release=True, overload_release=True)


@torch.inference_mode()
def suspended_equivalence(device="cuda:0", n=64, steps=64):
    """Exercise the connected two ropes and four blocks under the same pad trajectory."""
    device = wp.get_device(device)
    rng = np.random.default_rng(521)
    cfg = PayloadParamsX.from_presets(
        PAYLOAD, ["folded_clothes"] * n, fill_ratio=torch.full((n, 1), .38),
        total_mass=torch.tensor(rng.uniform(.20, .31, (n, 1)), dtype=torch.float32),
        bag_half_extents=torch.tensor(rng.uniform([.069, .081, .009], [.112, .131, .015], (n, 3)),
                                      dtype=torch.float32))
    model = UnifiedPayloadTemplate(cfg)
    values = dict(MONDAY["proxy"])
    nominal = ProxyParams.from_scalars(values)
    rp = replace(nominal, **{name: v.repeat(n, 1) for name, v in vars(nominal).items()})
    proxy = LumpedContactProxy(rp, gpu.DT)
    sim = MacroParcelSim(proxy, payload=model)
    state = sim.initial_state()
    trajectory_base = state.payload.pos.new_tensor(rng.uniform(
        [-.03, -.04, .08], [.03, .04, .10], (n, 1, 3)))
    start = state.payload._replace(pos=state.payload.pos + trajectory_base)
    state = state._replace(payload=start, com=model.com(start))
    state.contacts[:, :, 0] = .012
    state.contacts[:, :, 1] = 1.
    state.contacts[:, :, 3] = .10
    anchor, _ = model.anchors(start, sim.payload_mount_weights, torch.zeros_like(start.d_surf))
    extension = state.payload.pos.new_tensor(rng.uniform(.005, .015, (n, 1, 1)))
    tip = anchor + (rp.L0[:, :, None] + extension) * torch.tensor([0., 0., 1.])
    velocity = state.payload.pos.new_tensor(rng.uniform([-.035, -.025, .06], [.04, .05, .09], (n, 1, 3)))
    ts = gpu.DT * torch.arange(steps + 1, dtype=torch.float32)
    path = tip[:, None] + ts[None, :, None, None] * velocity[:, None]
    dev_path = wp.array(path.numpy(), dtype=wp.vec3, device=device)
    payload = wp.array([payload_parameters(model, i) for i in range(n)], dtype=gpu.PayloadParameters, device=device)
    ropes = wp.array([rope_parameters(values)] * n, dtype=gpu.RopeParameters, device=device)
    initial_blocks, initial_ropes, weights = [], [], []
    for i in range(n):
        b = gpu.PayloadState()
        b.position = gpu.Mat43(*start.pos[i].flatten())
        c = gpu.RopeState()
        c.contacts = gpu.Mat26(*state.contacts[i].flatten())
        c.reference = wp.vec2(*state.contact_reference[i])
        initial_blocks.append(b)
        initial_ropes.append(c)
        weights.append(gpu.Mat24(*sim.payload_mount_weights[i].flatten()))
    b = wp.array(initial_blocks, dtype=gpu.PayloadState, device=device)
    c = wp.array(initial_ropes, dtype=gpu.RopeState, device=device)
    mounts = wp.array(weights, dtype=gpu.Mat24, device=device)
    history = wp.empty((n, steps, 5), dtype=float, device=device)
    expected = []
    normal = torch.full((n, 2), 6.)
    zeros = torch.zeros_like(normal)
    for i in range(1, steps + 1):
        wp.launch(gpu.advance_suspended_reference, n, [dev_path, payload, ropes, b, c, mounts, i, history],
                  device=device)
        pd = path[:, i]
        action = ParcelAction(pd, (pd - path[:, i - 1]) / gpu.DT, normal, zeros, zeros, zeros,
                              torch.zeros((n, 3)), torch.zeros((n, 1)))
        result = sim.step(state, action)
        state = result.state
        expected.append(torch.cat((result.tension, state.contacts[:, :, 1], result.table_clearance), dim=1))
    actual, target = history.numpy(), torch.stack(expected, dim=1).numpy()
    np.testing.assert_allclose(actual[..., :2], target[..., :2], rtol=5e-4, atol=.003)
    np.testing.assert_array_equal(actual[..., 2:4], target[..., 2:4])
    np.testing.assert_allclose(actual[..., 4], target[..., 4], rtol=2e-3, atol=2e-5)
    return dict(device=device.name, independent_worlds=n, substeps=steps,
                max_tension_error_N=float(np.max(np.abs(actual[..., :2] - target[..., :2]))),
                max_clearance_error_m=float(np.max(np.abs(actual[..., 4] - target[..., 4]))),
                capture_steps_match=True, compared_design="Suspended shared submodel with optional neck and torsion disabled")


def evaluate(device="cuda:0", n=128):
    device = wp.get_device(device)
    if not device.is_cuda:
        raise ValueError("C0 GPU equivalence requires CUDA")
    rng = np.random.default_rng(490)
    return dict(device=device.name, independent_worlds=n, payload_substeps=40,
                rope_max_abs_error=_compare_ropes(device, n, rng),
                payload_max_abs_error=_compare_blocks(device, n, rng),
                strict_release=_strict_release(device),
                scope="Isolated rope law and four-block integration on complete shared inputs; "
                      "the coupled Yam_B dynamics and the legacy neck/torsion replay are separate gates.")
