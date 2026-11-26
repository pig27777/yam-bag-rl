"""GPU state sampling and coupled Yam_B pad / four-block integration."""

import warp as wp

from ..robot.yam_warp_kernels import Actuators, Commands, Geometry, coordinates
from .surrogate_kernels import (
    DT, EPS, Mat23, Mat43, Mat24, PayloadParameters, PayloadState, RopeInput, RopeParameters, RopeState,
    material_weights, normal_for, payload_contact, project_to_material, row2, row4, weighted,
)

wp.set_module_options({"enable_backward": False, "fuse_fp": False})
BAG_HALF = (.111, .083, .012)


@wp.struct
class World:
    payload: PayloadParameters
    ropes: RopeParameters
    blocks: PayloadState
    grip: RopeState
    mounts: Mat24
    previous_tips: Mat23
    has_tip: bool
    rearm_gap: float
    slip_limit: float
    center: wp.vec3
    initial_core_center: wp.vec3
    collision_velocity: float
    bin_min_clearance: float


@wp.struct
class Audit:
    invalid: int
    cutoff: int
    probe_unloaded: int
    probe_contact: int
    released: int
    slipped: int
    captured: int
    peak_clearance: float
    current_clearance: float
    last_speed: float
    direct_peak: float
    direct_now: float
    direct_integral: float
    peak_tension: float
    probe_effort_cost: float
    probe_samples: int
    peak_residual: float
    transport_speed: float
    support_N: float
    rope_up_N: float
    peak_solver_iterations: int
    retracted: int
    success: int
    overflow_flags: int
    peak_fold_N: float
    peak_feed_m: float
    peak_gather_m: float


@wp.struct
class Scene:
    state: wp.array[World]
    audits: wp.array[Audit]
    regime: wp.array[int]
    v0: wp.array2d[float]
    hidden: wp.array2d[float]
    stage: wp.array[int]
    collisions: wp.array[int]
    pad_bodies: wp.array[int]
    pad_offsets: wp.array[wp.vec3]
    jaw_effort: wp.array[float]


@wp.func
def uniform(rng: wp.uint32, low: float, high: float):
    return low + (high - low) * wp.randf(rng)


@wp.func
def positive_multiplier(rng: wp.uint32):
    return wp.exp(uniform(rng, -.805, .805))


@wp.kernel
def reset_parcel(scene: Scene, original: RopeParameters, seed: int, base_id: int, regimes: int, hidden_offset: int):
    i = wp.tid()
    identity = base_id + i
    rng = wp.rand_init(seed + hidden_offset, identity)
    obs_rng = wp.rand_init(seed + 80021, identity)
    regime = identity % 4
    # The public stream is separate from all hidden shape, mass and friction draws.
    center = wp.vec3(-.030 + uniform(obs_rng, -.006, .006),
                     -.225 + uniform(obs_rng, -.005, .005), 0.)
    v0_center = center + wp.vec3(.0004 * wp.randn(obs_rng), .0004 * wp.randn(obs_rng),
                                 .012 + .0002 * wp.randn(obs_rng))
    yaw = uniform(obs_rng, -.07, .07)
    c, s = wp.cos(yaw), wp.sin(yaw)
    for d in range(3):
        scene.v0[i, d] = v0_center[d]
    scene.v0[i, 3] = c
    scene.v0[i, 4] = s
    scene.v0[i, 5] = 0.
    scene.v0[i, 6] = -s
    scene.v0[i, 7] = c
    scene.v0[i, 8] = 0.
    scene.v0[i, 9] = .111 + .0003 * wp.randn(obs_rng)
    scene.v0[i, 10] = .083 + .0003 * wp.randn(obs_rng)
    scene.v0[i, 11] = .012 + .0002 * wp.randn(obs_rng)

    pack = uniform(rng, 0., .18)
    if regime == 2:
        pack = uniform(rng, .34, .69)
    elif regime == 3:
        pack = uniform(rng, .86, 1.)
    if regimes == 0:
        pack = wp.randf(rng)
    total_mass = uniform(rng, .150, .400)
    bias = uniform(rng, -.28, .28)
    if regime < 2:
        bias = float(2 * regime - 1) * uniform(rng, .78, 1.)
    bias *= 1. - pack
    fore_aft = uniform(rng, -.5, .5) * (1. - pack)
    soft = positive_multiplier(rng)
    p = PayloadParameters()
    p.half = wp.vec3(uniform(rng, .067, .103) + .014 * pack,
                     uniform(rng, .050, .070) + .011 * pack, .012)
    p.slide = uniform(rng, .0001, .0018) * (1. - pack)
    p.stretch_alpha = (.0016 * (1. - pack) + .000035) / (DT * DT) / soft
    p.bend_alpha = (.022 * (1. - pack) + .0003) / (DT * DT) / soft
    p.bend_limit = .5 * wp.length(wp.vec2(p.half[0], p.half[1])) * (1. - .92 * pack)
    p.damping = uniform(rng, .20, .83)
    p.friction = uniform(rng, .22, .64)
    p.rebound = wp.exp(-DT / uniform(rng, .25, .65))
    p.soft_k = 120. * soft
    p.hardening = 5.
    p.compression = 3.5 * positive_multiplier(rng)
    p.lateral_k = 220. * soft
    hidden_angle = yaw + uniform(rng, -.23, .23) * (1. - .5 * pack)
    ca, sa = wp.cos(hidden_angle), wp.sin(hidden_angle)
    offset = wp.vec3(.023 * bias, .016 * fore_aft, 0.)
    st = PayloadState()
    minimum = wp.vec4()
    shares = wp.vec4()
    for j in range(4):
        x, y = float(2 * (j % 2) - 1), float(2 * (j // 2) - 1)
        minimum[j] = uniform(rng, .005, .015)
        shares[j] = .25 * wp.max(0., (1. + x * bias) * (1. + y * fore_aft))
        # The heaviest blocks are thicker in loose fills. Flat-full blocks stay thin.
        height = .004 + .034 * shares[j] * (1. - pack) + .0035 * pack
        p.height[j] = height
        p.max_depth[j] = 2. * height * (.63 * (1. - pack) + .13 * pack)
        loc = wp.vec3(.5 * p.half[0] * x, .5 * p.half[1] * y, 0.)
        p.rest[j] = loc
        st.position[j] = center + offset + wp.vec3(ca * loc[0] - sa * loc[1], sa * loc[0] + ca * loc[1], height)
    free_mass = total_mass - wp.dot(minimum, wp.vec4(1.))
    for j in range(4):
        p.mass[j] = minimum[j] + free_mass * shares[j]
        p.inv_mass[j] = 1. / p.mass[j]
    rp = original
    rp.stiffness *= positive_multiplier(rng)
    rp.damping *= positive_multiplier(rng)
    rp.friction *= wp.exp(uniform(rng, -.58, .58))
    # Rest length increases by 20–50 mm in loose packs and 2–5 mm in flat-full packs.
    slack = uniform(rng, .020, .05)
    if regime == 2:
        slack = uniform(rng, .011, .022)
    if regime == 3:
        slack = uniform(rng, .002, .005)
    if regimes == 0:
        slack = (1. - pack) * slack + pack * .004
    rp.rest = wp.vec2(slack, slack * uniform(rng, .89, 1.11))
    # This reserve is the available gathered film, separate from the axial rest length.
    rp.budget *= uniform(rng, .75, 1.12)
    grip = RopeState()
    grip.reference = rp.rest
    world = World()
    world.payload = p
    world.ropes = rp
    world.blocks = st
    world.grip = grip
    world.mounts[0] = material_weights(p.half, wp.vec3(-.024, 0., 0.))
    world.mounts[1] = material_weights(p.half, wp.vec3(.024, 0., 0.))
    world.slip_limit = uniform(rng, .82, 1.15)
    world.center = center
    world.initial_core_center = weighted(wp.vec4(.25), st.position)
    world.rearm_gap = .024
    scene.state[i] = world
    scene.regime[i] = regime
    scene.audits[i] = Audit()
    scene.stage[i] = 0
    scene.jaw_effort[i] = .32
    scene.hidden[i, 0] = total_mass / .4
    scene.hidden[i, 1] = bias
    scene.hidden[i, 2] = pack
    scene.hidden[i, 3] = fore_aft
    scene.hidden[i, 4] = hidden_angle
    scene.hidden[i, 5] = p.friction
    scene.hidden[i, 6] = rp.stiffness / 150.
    scene.hidden[i, 7] = rp.damping / 8.
    scene.hidden[i, 8] = rp.rest[0] * 15.
    scene.hidden[i, 9] = rp.rest[1] * 15.
    scene.hidden[i, 10] = rp.friction[0]
    scene.hidden[i, 11] = p.soft_k / 200.
    scene.hidden[i, 12] = p.half[0] * 10.
    scene.hidden[i, 13] = p.half[1] * 10.


@wp.kernel
def couple_parcel(scene: Scene, robot: Geometry, cmd: Commands, motor: Actuators,
                  qpos: wp.array2d[float], x: wp.array2d[wp.vec3], rotation: wp.array2d[wp.mat33],
                  anchor: wp.array2d[wp.vec3], axes: wp.array2d[wp.vec3], applied: wp.array2d[float]):
    env = wp.tid()
    world = scene.state[env]
    audit = scene.audits[env]
    for j in range(applied.shape[1]):
        applied[env, j] = 0.
    if audit.invalid != 0:
        return
    y = RopeInput()
    tip_speed = Mat23()
    for j in range(2):
        body = scene.pad_bodies[j]
        y.tip[j] = x[env, body] + rotation[env, body] * scene.pad_offsets[j]
        if world.has_tip:
            tip_speed[j] = (row2(y.tip, j) - row2(world.previous_tips, j)) / DT
    y.tip_velocity = tip_speed
    world.previous_tips = y.tip
    world.has_tip = True
    separation = row2(y.tip, 1) - row2(y.tip, 0)
    rail = robot.joints[7]
    direction = axes[env, rail]
    gap = wp.max(0., wp.dot(separation, direction))
    delta_speed = row2(tip_speed, 1) - row2(tip_speed, 0)
    max_normal = wp.max(0., -motor.applied[env, 6])
    y.opening = cmd.requested[env, 6] > .038
    y.closing = cmd.requested[env, 6] < .021
    y.slip_limit = world.slip_limit
    y.budget = world.ropes.budget
    closing_speed = -.5 * wp.dot(delta_speed, direction)
    folded = wp.max(0., 800. * (world.rearm_gap - gap) + 7. * closing_speed)
    folded = wp.min(1.5 * max_normal, folded)
    if gap >= world.rearm_gap:
        folded = 0.
    film_normal = normal_for(world.blocks.position)
    film_center = weighted(wp.vec4(.25), world.blocks.position)
    outer_center = world.center + film_center - world.initial_core_center
    y.inward = wp.vec2(closing_speed)
    for j in range(2):
        p = row2(y.tip, j)
        local = project_to_material(p, world.blocks.position, film_normal)
        w = material_weights(world.payload.half, local)
        local_top = wp.dot(w, world.payload.height) + wp.dot(weighted(w, world.blocks.position) - film_center, film_normal)
        # Empty outer material keeps its nominal footprint and loose fold depth
        # even when the internal blocks lean or leave one side nearly empty.
        film_xy = p - outer_center
        feed = wp.max(local_top - local[2], .025 - film_xy[2])
        touch = wp.abs(film_xy[0]) <= .111 and wp.abs(film_xy[1]) <= .083 and feed > 0. and p[2] > .0004
        previous = world.grip.contacts[j, 1] > .5
        if (touch or previous) and not y.opening:
            y.normal[j] = folded
        if touch:
            y.feed[j] = feed
            y.alignment[j] = wp.sqrt(wp.max(0., 1. - wp.pow(wp.dot(direction, film_normal), 2.)))
        audit.peak_fold_N = wp.max(audit.peak_fold_N, y.normal[j])
        audit.peak_feed_m = wp.max(audit.peak_feed_m, y.feed[j])
    result = payload_contact(world.payload, world.ropes, world.blocks, world.grip, y, world.mounts, True)
    stage = scene.stage[env]
    if (stage == 2 or stage == 3) and (
            wp.length(row2(result.reaction, 0)) + wp.length(row2(result.reaction, 1)) > .12):
        audit.probe_contact = 1
    for j in range(2):
        if world.grip.contacts[j, 1] > .5 and result.ropes.state.contacts[j, 1] < .5:
            audit.released += 1
            if result.ropes.state.slipped[j] != 0:
                audit.slipped += 1
    world.grip = result.ropes.state
    audit.peak_gather_m = wp.max(audit.peak_gather_m, wp.max(world.grip.contacts[0, 0], world.grip.contacts[1, 0]))
    old_positions = world.blocks.position
    world.blocks = result.blocks.state
    audit.captured = int(world.grip.contacts[0, 1]) + int(world.grip.contacts[1, 1])
    audit.direct_now = result.direct[0] + result.direct[1]
    audit.direct_peak = wp.max(audit.direct_peak, audit.direct_now)
    audit.direct_integral += DT * audit.direct_now
    audit.current_clearance = 100.
    audit.last_speed = 0.
    audit.support_N = wp.dot(result.blocks.support, wp.vec4(1.))
    audit.rope_up_N = result.ropes.force[0, 2] + result.ropes.force[1, 2]
    audit.peak_tension = wp.max(audit.peak_tension, wp.max(result.ropes.tension[0], result.ropes.tension[1]))

    inside_bin = True
    for j in range(4):
        pos = row4(world.blocks.position, j)
        v = row4(world.blocks.velocity, j)
        # Native bin floor is at 4 mm. Each block has a support footprint; count
        # a deposit only when all blocks have crossed the rim with margin.
        inside = wp.abs(pos[0] + .03) < .197 and wp.abs(pos[1] - .03) < .138
        prior_pos = row4(old_positions, j)
        old_inside = wp.abs(prior_pos[0] + .03) < .197 and wp.abs(prior_pos[1] - .03) < .138
        h = world.payload.height[j] - .5 * world.blocks.depth[j]
        if inside and not old_inside and pos[2] - h < .122:
            # Inelastic surrogate wall constraint, not an attachment.
            pos = wp.vec3(prior_pos[0], prior_pos[1], pos[2])
            v = wp.vec3(0., 0., v[2])
            inside = False
        if inside:
            pos[2] = wp.max(pos[2], h + .004)
            if pos[2] <= h + .004:
                normal_dv = wp.max(DT * 9.81, -v[2])
                planar_speed = wp.length(wp.vec2(v[0], v[1]))
                fraction = wp.max(0., 1. - world.payload.friction * normal_dv / wp.max(planar_speed, EPS))
                v[0] *= fraction
                v[1] *= fraction
                v[2] = wp.max(0., v[2])
        inside_bin = inside_bin and inside
        world.blocks.position[j] = pos
        world.blocks.velocity[j] = v
        audit.current_clearance = wp.min(audit.current_clearance, result.blocks.clearance[j])
        audit.last_speed = wp.max(audit.last_speed, wp.length(v))
        if not wp.isfinite(wp.length(pos)) or not wp.isfinite(wp.length(v)) or wp.length(v) > 8. or wp.length(pos) > 2.:
            audit.invalid = 2
    audit.peak_clearance = wp.max(audit.peak_clearance, audit.current_clearance)
    if stage == 8 or stage == 9:
        audit.transport_speed = wp.max(audit.transport_speed, audit.last_speed)
    if stage >= 10:
        audit.success = int(inside_bin and audit.captured == 0 and audit.current_clearance < .012 and
                            audit.peak_clearance > .125 and audit.last_speed < .020)
    if audit.peak_tension > 100. or not wp.isfinite(audit.peak_tension):
        audit.invalid = 3
    # At the frozen core spring scale this represents severe penetration or an
    # unstable encounter, beyond the range of a valid compliant press.
    if audit.direct_now > 120. or not wp.isfinite(audit.direct_now):
        audit.invalid = 4
    if audit.invalid == 0:
        for j in range(2):
            fold = y.normal[j]
            if j == 0:
                fold = -fold
            force = row2(result.reaction, j) + fold * direction
            tip = row2(y.tip, j)
            for k in range(6):
                id = robot.joints[k]
                jac = wp.cross(axes[env, id], tip - anchor[env, id])
                applied[env, robot.dof[k]] += wp.dot(jac, force)
            finger = robot.joints[6 + j]
            applied[env, robot.dof[6 + j]] += wp.dot(axes[env, finger], force)
    scene.state[env] = world
    scene.audits[env] = audit
