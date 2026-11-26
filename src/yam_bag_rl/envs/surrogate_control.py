"""GPU macro command, deployable telemetry, and safety kernels for the surrogate."""

import warp as wp

from ..robot.yam_warp_kernels import Actuators, Commands, Geometry, Mat66, Vec6, coordinates, damped_solve
from .surrogate_scene import Scene, uniform

wp.set_module_options({"enable_backward": False})
SENSOR_FIELDS = 62
PRESS = 2
SWEEP = 3
RETRACT = 4
GRASP_PRE = 5
GRASP_DOWN = 6
GRASP_CLOSE = 7
LIFT = 8
TRANSPORT = 9
RELEASE = 10
SETTLE = 11
HARD_OVERFLOWS = (1 << 9) - 1


@wp.struct
class Motion:
    actions: wp.array2d[float]
    candidates: wp.array3d[float]
    selected: wp.array[int]
    target_position: wp.array[wp.vec3]
    target_rotation: wp.array[wp.quat]
    unload: wp.array2d[float]
    hover_q: wp.array[float]
    hover_xyz: wp.vec3
    hover_quat: wp.quat
    lo: wp.vec3
    hi: wp.vec3
    skip_try: wp.array[int]
    fault: wp.array[int]


@wp.struct
class Telemetry:
    filters: wp.array2d[float]
    gain: wp.array2d[float]
    bias: wp.array2d[float]
    noise: wp.array2d[float]
    alpha: wp.array2d[float]
    deadband: wp.array2d[float]
    dropout: wp.array2d[int]
    sensor_delay: wp.array[int]
    control_delay: wp.array[int]
    commands: wp.array3d[float]
    ring: wp.array3d[float]
    history: wp.array3d[float]
    valid: wp.array2d[bool]
    windows: wp.array2d[int]
    stamps: wp.array2d[float]
    nominal_mass: wp.array[float]
    descendants: wp.array2d[int]
    counter: wp.array[int]
    identity: wp.array[int]


@wp.func
def public_axis(v: wp.array2d[float], env: int, index: int):
    return wp.vec3(v[env, index], v[env, index + 1], v[env, index + 2])


@wp.func
def in_frame(u: wp.vec3, x: wp.vec3, y: wp.vec3):
    return u[0] * x + u[1] * y + u[2] * wp.cross(x, y)


@wp.func
def rot_error(target: wp.quat, current: wp.quat):
    delta = wp.normalize(target * wp.quat_inverse(current))
    v = wp.vec3(delta[0], delta[1], delta[2])
    if delta[3] < 0.:
        v = -v
    magnitude = wp.length(v)
    return v * (2. * wp.atan2(magnitude, wp.abs(delta[3])) / wp.max(1e-9, magnitude))


@wp.func
def axes_to_quaternion(x: wp.vec3, y: wp.vec3):
    x1 = wp.normalize(x)
    y1 = wp.normalize(y - wp.dot(x1, y) * x1)
    z = wp.cross(x1, y1)
    return wp.quat_from_matrix(wp.matrix_from_cols(x1, y1, z))


@wp.kernel
def sample_candidates(v0: wp.array2d[float], seed: int, identity: wp.array[int],
                      hover: wp.quat, mode: int, output: wp.array3d[float]):
    i, k = wp.tid()
    rng = wp.rand_init(seed + k * 8191, identity[i])
    home_x = wp.quat_rotate(hover, wp.vec3(1., 0., 0.))
    home_y = wp.quat_rotate(hover, wp.vec3(0., 1., 0.))
    fx = public_axis(v0, i, 3)
    fy = public_axis(v0, i, 6)
    size = public_axis(v0, i, 9)
    gripper = uniform(rng, .062, .090)
    effort = uniform(rng, .42, .70)
    x = float(0.)
    y = float(0.)
    z = float(0.)
    yaw = float(0.)
    roll = float(0.)
    pitch = float(0.)
    if mode == 1 and k >= output.shape[1] // 2:
        x = size[0] * uniform(rng, -.78, .78)
        y = size[1] * uniform(rng, -.62, .62)
        z = uniform(rng, .013, .037)
        yaw = uniform(rng, -.80, .80)
        roll = uniform(rng, -.16, .16)
        pitch = uniform(rng, -.16, .16)
    else:
        # Sixteen geometric templates. Their continuous noise is independent of the hidden state.
        template = k % 16
        x = size[0] * (.36 * float(template % 4) - .54) + uniform(rng, -.011, .011)
        y = size[1] * (.34 * float((template // 4) % 4) - .51) + uniform(rng, -.009, .009)
        z = .021 + uniform(rng, -.003, .003)
        yaw = uniform(rng, -.13, .13) + .29 * float((template + 1) % 3 - 1)
        if template >= 12:
            roll = .12 * float((template % 2) * 2 - 1)
            pitch = uniform(rng, -.07, .07)
        if template == 0 or template == 3:
            x *= 1.3
    yaw += wp.atan2(fx[1], fx[0])
    q = (wp.quat_from_axis_angle(wp.vec3(0., 0., 1.), yaw) *
         wp.quat_from_axis_angle(wp.vec3(1., 0., 0.), roll) *
         wp.quat_from_axis_angle(wp.vec3(0., 1., 0.), pitch))
    a = wp.quat_rotate(q, home_x)
    b = wp.quat_rotate(q, home_y)
    frame = wp.matrix_from_cols(fx, fy, wp.cross(fx, fy))
    local_rotation_a = wp.transpose(frame) * a
    local_rotation_b = wp.transpose(frame) * b
    output[i, k, 0] = x
    output[i, k, 1] = y
    output[i, k, 2] = z - v0[i, 2]
    for j in range(3):
        output[i, k, 3 + j] = local_rotation_a[j]
        output[i, k, 6 + j] = local_rotation_b[j]
    output[i, k, 9] = gripper
    output[i, k, 10] = effort


@wp.kernel
def prepare_phase(scene: Scene, motion: Motion, phase: int, bins: wp.vec3, corridor: bool):
    i = wp.tid()
    scene.stage[i] = phase
    v0 = scene.v0
    a = motion.actions
    fx = public_axis(v0, i, 3)
    fy = public_axis(v0, i, 6)
    center = public_axis(v0, i, 0)
    ext = public_axis(v0, i, 9)
    p = motion.hover_xyz
    q = motion.hover_quat
    width = .080
    effort = .32
    work = scene.audits[i]
    if (phase >= 1 and phase <= SWEEP) and motion.skip_try[i] == 0:
        start = wp.vec3(.68 * ext[0] * a[i, 0], .61 * ext[1] * a[i, 1], ext[2] + .012 + .006 * a[i, 2])
        dp = wp.vec3(.019 * a[i, 6], .019 * a[i, 7], -.025 + .013 * a[i, 8])
        sweep = wp.vec3(.030 * a[i, 9], .030 * a[i, 10], .011 * a[i, 11])
        if phase == PRESS or phase == SWEEP:
            start += dp
            width = .064 + .015 * a[i, 12]
        if phase == SWEEP:
            start += sweep
        p = center + in_frame(start, fx, fy)
        rotation = in_frame(.32 * wp.vec3(a[i, 3], a[i, 4], a[i, 5]), fx, fy)
        angle = wp.length(rotation)
        if angle > 1e-7:
            q = wp.quat_from_axis_angle(rotation / angle, angle) * q
    if phase >= GRASP_PRE and phase <= RELEASE:
        k = motion.selected[i]
        pos = wp.vec3(motion.candidates[i, k, 0], motion.candidates[i, k, 1], motion.candidates[i, k, 2])
        g1 = wp.vec3(motion.candidates[i, k, 3], motion.candidates[i, k, 4], motion.candidates[i, k, 5])
        g2 = wp.vec3(motion.candidates[i, k, 6], motion.candidates[i, k, 7], motion.candidates[i, k, 8])
        q = axes_to_quaternion(in_frame(g1, fx, fy), in_frame(g2, fx, fy))
        p = center + in_frame(pos, fx, fy)
        width = wp.clamp(motion.candidates[i, k, 9], .055, .095)
        effort = wp.clamp(motion.candidates[i, k, 10], .05, .70)
        if phase == GRASP_PRE:
            p -= .046 * wp.quat_rotate(q, wp.vec3(0., 0., 1.))
        if phase >= GRASP_CLOSE and phase < RELEASE:
            width = .004
        if phase >= LIFT:
            p[2] = .170
        if phase == LIFT and corridor:
            p[0] = bins[0]
            p[2] = .304
            q = wp.quat_from_axis_angle(wp.vec3(0., 1., 0.), -.74) * motion.hover_quat
            if motion.fault[i] == 1:
                effort = .002
            elif motion.fault[i] == 2:
                # Reach the same safe lift corridor via an eccentric diagonal.
                p[0] -= .047
                p[1] -= .065
        if phase >= TRANSPORT:
            p[0] = bins[0]
            p[1] = bins[1]
            q = wp.quat_from_axis_angle(wp.vec3(0., 1., 0.), -.74) * motion.hover_quat
            p[2] = .304
        if phase == RELEASE:
            width = .080
        if work.cutoff == 2:
            p = motion.hover_xyz
            q = motion.hover_quat
            width = .080
    if phase == SETTLE:
        p = wp.vec3(bins[0], bins[1], .304)
        q = wp.quat_from_axis_angle(wp.vec3(0., 1., 0.), -.74) * motion.hover_quat
    if phase == 0 or phase == RETRACT or phase >= SETTLE:
        width = .080
    motion.target_position[i] = wp.vec3(wp.clamp(p[0], motion.lo[0], motion.hi[0]),
                                         wp.clamp(p[1], motion.lo[1], motion.hi[1]),
                                         wp.clamp(p[2], motion.lo[2], motion.hi[2]))
    motion.target_rotation[i] = q
    scene.jaw_effort[i] = effort
    motion.unload[i, 6] = width


@wp.kernel
def configure_faults(scene: Scene, modes: wp.array[int]):
    i = wp.tid()
    if modes[i] == 2:
        state = scene.state[i]
        state.slip_limit = .21 + .022 * float(i % 8)
        scene.state[i] = state


@wp.kernel
def audit_collision_depth(count: wp.array[int], worlds: wp.array[int], pairs: wp.array[wp.vec2i],
                          depth: wp.array[float], groups: wp.array[int], flags: wp.array[int]):
    j = wp.tid()
    if j < count[0]:
        pair = pairs[j]
        if pair[0] >= 0 and pair[1] >= 0:
            a, b = groups[pair[0]], groups[pair[1]]
            fixed_base_on_table = (a == 2 and b == 3) or (a == 3 and b == 2)
            arm_collision = (a == 1 or b == 1) and not fixed_base_on_table
            if arm_collision and depth[j] < -.004:
                severe = depth[j] < -.018
                wp.atomic_max(flags, worlds[j], 1 + int(severe))


@wp.func
def differential_ik(robot: Geometry, i: int, point: wp.vec3, xyz: wp.vec3, rotation: wp.vec3,
                    anchors: wp.array2d[wp.vec3], axes: wp.array2d[wp.vec3]):
    jac = Mat66()
    for j in range(6):
        angular = axes[i, robot.joints[j]]
        linear = wp.cross(angular, point - anchors[i, robot.joints[j]])
        for k in range(3):
            jac[k, j] = 4. * linear[k]
            jac[3 + k, j] = angular[k]
    return damped_solve(jac, Vec6(4. * xyz[0], 4. * xyz[1], 4. * xyz[2],
                                  rotation[0], rotation[1], rotation[2]), .028)


@wp.kernel
def stage_command(scene: Scene, motion: Motion, motor: Actuators, robot: Geometry, cmd: Commands,
                  qpos: wp.array2d[float], site: wp.array2d[wp.vec3], site_r: wp.array2d[wp.mat33],
                  anchors: wp.array2d[wp.vec3], axes: wp.array2d[wp.vec3]):
    i = wp.tid()
    phase = scene.stage[i]
    audit = scene.audits[i]
    point = site[i, robot.site]
    orientation = wp.quat_from_matrix(site_r[i, robot.site])
    dx = motion.target_position[i] - point
    ang = rot_error(motion.target_rotation[i], orientation)
    dx *= wp.min(1., .033 / wp.max(wp.length(dx), 1e-8))
    ang *= wp.min(1., .20 / wp.max(wp.length(ang), 1e-8))
    dq = differential_ik(robot, i, point, dx, ang, anchors, axes)
    for j in range(6):
        q = qpos[i, robot.pos[j]]
        target = q + wp.clamp(dq[j], -.12, .12)
        if phase == 0 or phase == RETRACT or (phase <= SWEEP and motion.skip_try[i] != 0):
            target = motion.hover_q[j]
        if phase <= SWEEP and audit.cutoff == 1:
            target = motion.unload[i, j]
        if audit.invalid != 0 or (audit.cutoff == 2 and phase >= GRASP_PRE):
            target = q
        cmd.requested[i, j] = wp.clamp(target, robot.lower[j], robot.upper[j])
        cmd.velocity[i, j] = 0.
        cmd.feedforward[i, j] = 0.
        cmd.action[i, j] = cmd.requested[i, j]
    gap = motion.unload[i, 6]
    if (phase <= SWEEP and audit.cutoff == 1) or audit.invalid != 0 or audit.cutoff == 2:
        gap = .080
    cmd.requested[i, 6] = gap
    cmd.velocity[i, 6] = 0.
    cmd.feedforward[i, 6] = 0.
    cmd.action[i, 6] = gap
    tuning = motor.tuning[i, 6]
    tuning.limit = scene.jaw_effort[i] / robot.motor_Nm_per_N
    motor.tuning[i, 6] = tuning


@wp.kernel
def reset_telemetry(t: Telemetry, seed: int, base_id: int, training: bool):
    i = wp.tid()
    identity = base_id + i
    t.identity[i] = identity
    rng = wp.rand_init(seed, identity)
    t.counter[i] = 0
    t.sensor_delay[i] = int(wp.randi(rng, 2, 11))
    t.control_delay[i] = int(wp.randi(rng, 1, 6))
    for j in range(7):
        t.filters[i, j] = 0.
        t.gain[i, j] = uniform(rng, .91, 1.10)
        scale = float(1.)
        if j == 6:
            scale = .19
        t.bias[i, j] = scale * uniform(rng, -.050, .050)
        t.noise[i, j] = scale * uniform(rng, .010, .028)
        t.alpha[i, j] = 1. - wp.exp(-.002 / uniform(rng, .009, .022))
        t.deadband[i, j] = scale * uniform(rng, .012, .030)
        t.dropout[i, j] = int(training and wp.randf(rng) < .20)
    for f in range(t.history.shape[1]):
        t.valid[i, f] = False
        t.windows[i, f] = 0
        t.stamps[i, f] = 0.
        for j in range(t.history.shape[2]):
            t.history[i, f, j] = 0.


@wp.kernel
def command_delay(t: Telemetry, cmd: Commands, delayed_target: wp.array2d[float]):
    i = wp.tid()
    step = t.counter[i]
    slot = step % t.commands.shape[1]
    delayed = (step - t.control_delay[i] + t.commands.shape[1]) % t.commands.shape[1]
    for j in range(7):
        t.commands[i, slot, j] = cmd.target[i, j]
        command = cmd.target[i, j]
        if step >= t.control_delay[i]:
            command = t.commands[i, delayed, j]
        delayed_target[i, j] = command


@wp.kernel
def sense_and_guard(scene: Scene, motion: Motion, t: Telemetry, motor: Actuators, robot: Geometry,
                    cmd: Commands, qpos: wp.array2d[float], qvel: wp.array2d[float],
                    xipos: wp.array2d[wp.vec3], anchors: wp.array2d[wp.vec3], axes: wp.array2d[wp.vec3],
                    sites: wp.array2d[wp.vec3], site_r: wp.array2d[wp.mat33],
                    actuator_force: wp.array2d[float], overflow: wp.array[int], solver_niter: wp.array[int],
                    stamp: wp.array[float], seed: int):
    i = wp.tid()
    audit = scene.audits[i]
    phase = scene.stage[i]
    step = t.counter[i]
    slot = step % t.ring.shape[1]
    t.counter[i] = step + 1
    current = wp.quat_from_matrix(site_r[i, robot.site])
    p = sites[i, robot.site]
    pos_error = motion.target_position[i] - p
    ori_error = rot_error(motion.target_rotation[i], current)
    res_peak = float(0.)
    tracking_peak = float(0.)
    effort_cost = float(0.)
    grip_effort = float(0.)
    audit.peak_solver_iterations = wp.max(audit.peak_solver_iterations, solver_niter[i])
    if (overflow[i] & HARD_OVERFLOWS) != 0:
        audit.invalid = 1
        audit.overflow_flags |= overflow[i]
    if scene.collisions[i] >= 2:
        audit.invalid = 5
    for j in range(7):
        pos = coordinates(qpos, robot.pos, i, j)
        vel = coordinates(qvel, robot.dof, i, j)
        sent = cmd.target[i, j]
        force = actuator_force[i, robot.actuators[j]]
        if j == 6:
            force *= robot.motor_Nm_per_N
        rng = wp.rand_init(seed + 1897 * j, t.identity[i] + step * 39551)
        raw = t.gain[i, j] * force + t.bias[i, j] + t.noise[i, j] * wp.randn(rng)
        if wp.abs(raw) < t.deadband[i, j]:
            raw = 0.
        effort = t.filters[i, j] + t.alpha[i, j] * (raw - t.filters[i, j])
        if step == 0:
            effort = raw
        t.filters[i, j] = effort
        if not wp.isfinite(pos) or not wp.isfinite(vel) or not wp.isfinite(effort) or wp.abs(vel) > 30.:
            audit.invalid = 2
        if j < 6:
            id = robot.joints[j]
            gravity = float(0.)
            # Frozen nominal model with zero velocity. Its generalized gravity bias is -J^T mg.
            for index in range(t.descendants.shape[1]):
                body = t.descendants[j, index]
                if body >= 0:
                    lever = xipos[i, body] - anchors[i, id]
                    gravity -= wp.dot(axes[i, id], wp.cross(lever, robot.gravity * t.nominal_mass[body]))
            residual = effort - gravity
            res_peak = wp.max(res_peak, wp.abs(residual))
            tracking_peak = wp.max(tracking_peak, wp.abs(sent - pos))
            effort_cost += wp.min(1., wp.abs(residual) / 3.)
            t.ring[i, slot, j] = pos
            t.ring[i, slot, j + 6] = vel
            t.ring[i, slot, j + 12] = sent
            t.ring[i, slot, j + 18] = sent - pos
            t.ring[i, slot, j + 24] = effort * float(t.dropout[i, j] == 0)
            t.ring[i, slot, j + 30] = residual * float(t.dropout[i, j] == 0)
        else:
            grip_effort = wp.abs(effort)
            t.ring[i, slot, 36] = wp.clamp(pos, 0., robot.max_gap)
            t.ring[i, slot, 37] = vel
            t.ring[i, slot, 38] = effort * float(t.dropout[i, j] == 0)
            t.ring[i, slot, 39] = sent - pos
        t.ring[i, slot, 55 + j] = sent
    a = wp.quat_rotate(current, wp.vec3(1., 0., 0.))
    b = wp.quat_rotate(current, wp.vec3(0., 1., 0.))
    for j in range(3):
        t.ring[i, slot, 40 + j] = p[j]
        t.ring[i, slot, 43 + j] = a[j]
        t.ring[i, slot, 46 + j] = b[j]
        t.ring[i, slot, 49 + j] = pos_error[j]
        t.ring[i, slot, 52 + j] = ori_error[j]
    if (phase == PRESS or phase == SWEEP) and motion.skip_try[i] == 0:
        audit.probe_samples += 1
        audit.probe_effort_cost += effort_cost / 6.
        audit.peak_residual = wp.max(audit.peak_residual, res_peak)
        if audit.cutoff == 0 and (res_peak > 2.2 or grip_effort > .33 or tracking_peak > .21):
            audit.cutoff = 1
            audit.probe_unloaded = 1
            dq = differential_ik(robot, i, p, wp.vec3(0., 0., .009), wp.vec3(), anchors, axes)
            for j in range(6):
                motion.unload[i, j] = wp.clamp(qpos[i, robot.pos[j]] + dq[j], robot.lower[j], robot.upper[j])
                cmd.requested[i, j] = motion.unload[i, j]
            cmd.requested[i, 6] = .080
    if phase >= GRASP_DOWN and phase <= TRANSPORT and (res_peak > 5.5 or grip_effort > .77 or tracking_peak > .32):
        audit.cutoff = 2
        cmd.requested[i, 6] = .080
    if phase == RETRACT:
        error = float(0.)
        speed = float(0.)
        for j in range(6):
            error = wp.max(error, wp.abs(qpos[i, robot.pos[j]] - motion.hover_q[j]))
            speed = wp.max(speed, wp.abs(qvel[i, robot.dof[j]]))
        audit.retracted = int(error < .038 and speed < .14 and
                              coordinates(qpos, robot.pos, i, 6) > .063)
    scene.audits[i] = audit


@wp.kernel
def record_history(scene: Scene, motion: Motion, telemetry: Telemetry, frame: int, dt: float):
    i = wp.tid()
    step = telemetry.counter[i] - 1
    delay = telemetry.sensor_delay[i]
    slot = (wp.max(0, step - delay)) % telemetry.ring.shape[1]
    telemetry.valid[i, frame] = motion.skip_try[i] == 0 and scene.audits[i].invalid == 0
    telemetry.windows[i, frame] = scene.stage[i]
    telemetry.stamps[i, frame] = dt * float(wp.max(0, step - delay) + 1)
    for j in range(telemetry.history.shape[2]):
        telemetry.history[i, frame, j] = telemetry.ring[i, slot, j]
