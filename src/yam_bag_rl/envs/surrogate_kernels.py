"""Warp implementation of the Monday dual-tether contact and four-point XPBD step."""

import warp as wp

# The reference uses separate tensor operations. Contraction can change the
# exact floor-contact predicate by one float32 unit after a position clamp.
wp.set_module_options({"enable_backward": False, "fuse_fp": False})
DT = .002
EPS = 1e-8
Vec6 = wp.types.vector(6, float)
Mat43 = wp.types.matrix((4, 3), float)
Mat23 = wp.types.matrix((2, 3), float)
Mat26 = wp.types.matrix((2, 6), float)
Mat24 = wp.types.matrix((2, 4), float)


@wp.struct
class RopeParameters:
    budget: float
    max_slack: float
    gather_fraction: float
    stiffness: float
    damping: float
    friction: wp.vec2
    slip_coefficient: float
    capture_slack: float
    gather_normal: float
    capture_normal: float
    release_normal: float
    capture_depth: float
    capture_alignment: float
    gather_delay: float
    gather_speed: float
    rest: wp.vec2


@wp.struct
class RopeInput:
    tip: Mat23
    anchor: Mat23
    tip_velocity: Mat23
    anchor_velocity: Mat23
    normal: wp.vec2
    inward: wp.vec2
    feed: wp.vec2
    alignment: wp.vec2
    mobility: wp.vec2
    budget: float
    opening: bool
    closing: bool
    slip_limit: float


@wp.struct
class RopeState:
    contacts: Mat26
    reference: wp.vec2
    # A forced slip requires leaving contact or opening before another acquisition.
    slipped: wp.vec2i


@wp.struct
class RopeResult:
    state: RopeState
    tension: wp.vec2
    force: Mat23
    slip_rate: wp.vec2
    energy: wp.vec2
    elastic: wp.vec2


@wp.func
def row2(a: Mat23, i: int):
    return wp.vec3(a[i, 0], a[i, 1], a[i, 2])


@wp.func
def row4(a: Mat43, i: int):
    return wp.vec3(a[i, 0], a[i, 1], a[i, 2])


@wp.func
def dot4(a: wp.vec4, b: wp.vec4):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]


# The CPU reference's contiguous three-value norm accumulates via FMA.
@wp.func_native("return sqrtf(fmaf(a[2], a[2], fmaf(a[1], a[1], a[0] * a[0])));")
def reference_norm(a: wp.vec3) -> float: ...


@wp.func
def step_rope(p: RopeParameters, old: RopeState, x: RopeInput, strict: bool):
    available = wp.vec2()
    holds = wp.vec2()
    fraction = wp.max(0., 1. - (old.contacts[0, 0] + old.contacts[1, 0]) / wp.max(x.budget, EPS))
    for j in range(2):
        if x.normal[j] >= p.gather_normal:
            holds[j] = old.contacts[j, 3] + DT
        gather = float(0.)
        if holds[j] >= p.gather_delay:
            gather = p.gather_fraction * wp.clamp(x.inward[j], 0., p.gather_speed) * fraction * DT
        available[j] = wp.min(old.contacts[j, 0] + gather, p.max_slack)
    available *= wp.min(1., x.budget / wp.max(available[0] + available[1], EPS))
    result = RopeResult()
    next_state = RopeState()
    for j in range(2):
        active = old.contacts[j, 1] > .5
        delta = row2(x.tip, j) - row2(x.anchor, j)
        distance = wp.length(delta)
        direction = delta / wp.max(distance, EPS)
        velocity = row2(x.tip_velocity, j) - row2(x.anchor_velocity, j)
        rate = wp.dot(velocity, direction)
        normal = wp.max(x.normal[j], 0.)
        damping = p.damping / (1. + DT * p.damping * x.mobility[j])
        before = distance - old.contacts[j, 2] - old.reference[j]
        trial = float(0.)
        if active and (before > 0. or (not strict and before == 0.)):
            trial = wp.max(0., p.stiffness * before + damping * rate)
        tangent = wp.length(velocity - rate * direction)
        # The Coulomb bound applies to axial pull and tangential impulse together.
        demand = wp.length(wp.vec2(trial, damping * tangent))
        forced = strict and active and (x.opening or normal < p.capture_normal or
                                        demand > x.slip_limit * p.friction[j] * normal)
        blocked = old.slipped[j] != 0
        if x.opening or (not active and x.feed[j] <= 0.):
            blocked = False
        if forced and not x.opening:
            blocked = True
        withdrawn = wp.min(available[j], DT * p.slip_coefficient * wp.max(0., trial - p.friction[j] * normal))
        slack = available[j] - withdrawn
        slip = old.contacts[j, 2] + withdrawn
        captured = False
        if active:
            captured = slack > EPS and normal >= p.release_normal and not forced
        else:
            captured = (x.feed[j] >= p.capture_depth and x.alignment[j] >= p.capture_alignment and
                        slack >= p.capture_slack and normal >= p.capture_normal)
            if strict:
                captured = captured and x.closing and not x.opening and not blocked
        if strict and (forced or x.opening):
            slack = 0.
            slip = 0.
        reference = p.rest[j]
        if captured:
            reference = wp.max(p.rest[j], distance - slip)
            if active:
                reference = old.reference[j]
        extension = distance - slip - reference
        tension = float(0.)
        if active and captured and (extension > 0. or (not strict and extension == 0.)):
            tension = wp.max(0., p.stiffness * extension + damping * (rate - withdrawn / DT))
        next_state.contacts[j, 0] = slack
        next_state.contacts[j, 1] = float(captured)
        next_state.contacts[j, 2] = slip
        next_state.contacts[j, 3] = holds[j]
        next_state.contacts[j, 4] = wp.max(x.feed[j], 0.)
        next_state.contacts[j, 5] = 0.
        if captured:
            peak = float(0.)
            if active:
                peak = old.contacts[j, 5]
            next_state.contacts[j, 5] = wp.max(tension, peak)
        next_state.reference[j] = reference
        next_state.slipped[j] = int(blocked)
        result.tension[j] = tension
        result.force[j] = tension * direction
        result.slip_rate[j] = withdrawn / DT
        if active and captured:
            extension = wp.max(0., extension)
            result.energy[j] = .5 * p.stiffness * extension * extension
            result.elastic[j] = p.stiffness * extension
    result.state = next_state
    return result


@wp.struct
class PayloadParameters:
    half: wp.vec3
    height: wp.vec4
    rest: Mat43
    mass: wp.vec4
    inv_mass: wp.vec4
    slide: float
    stretch_alpha: float
    bend_alpha: float
    bend_limit: float
    damping: float
    friction: float
    rebound: float
    max_depth: wp.vec4
    soft_k: float
    hardening: float
    compression: float
    lateral_k: float


@wp.struct
class PayloadState:
    position: Mat43
    velocity: Mat43
    depth: wp.vec4


@wp.struct
class PayloadResult:
    state: PayloadState
    support: wp.vec4
    clearance: wp.vec4


@wp.func
def edge_a(i: int):
    return (i % 2) * wp.max(1, 2 - i // 2)


@wp.func
def edge_b(i: int):
    j = i // 2
    if i % 2 == 0:
        return 1 + j
    return wp.min(3, 4 - j)


@wp.func
def advance_payload(p: PayloadParameters, old: PayloadState, force: Mat43, depth: wp.vec4):
    position = Mat43()
    normal_shift = wp.vec4()
    minimum = p.height - .5 * depth
    for j in range(4):
        position[j] = row4(old.position, j) + DT * row4(old.velocity, j) + .000004 * (
            p.inv_mass[j] * row4(force, j) - wp.vec3(0., 0., 9.81))
    multipliers = Vec6()
    bend_multiplier = float(0.)
    signs = wp.vec4(.5, -.5, -.5, .5)
    total = .25 * wp.dot(p.inv_mass, wp.vec4(1.))
    for iteration in range(4):
        for e in range(6):
            a, b = edge_a(e), edge_b(e)
            delta = row4(position, a) - row4(position, b)
            length = reference_norm(delta)
            error = length - reference_norm(row4(p.rest, a) - row4(p.rest, b))
            constraint = wp.sign(error) * wp.max(0., wp.abs(error) - p.slide)
            if p.slide == 0. or constraint != 0.:
                dl = (-constraint - p.stretch_alpha * multipliers[e]) / (p.inv_mass[a] + p.inv_mass[b] + p.stretch_alpha)
                correction = delta * (dl / wp.max(length, EPS))
                position[a] = row4(position, a) + p.inv_mass[a] * correction
                position[b] = row4(position, b) - p.inv_mass[b] * correction
                multipliers[e] += dl
            else:
                multipliers[e] = 0.
        bend = .5 * ((row4(position, 0) + row4(position, 3)) - (row4(position, 1) + row4(position, 2)))
        length = reference_norm(bend)
        soft = (-length - p.bend_alpha * bend_multiplier) / (total + p.bend_alpha)
        hard = -wp.max(0., length + total * soft - p.bend_limit) / total
        correction = bend * ((soft + hard) / wp.max(length, EPS))
        bend_multiplier += soft
        for j in range(4):
            point = row4(position, j) + correction * (p.inv_mass[j] * signs[j])
            normal_shift[j] += wp.max(0., minimum[j] - point[2])
            point[2] = wp.max(point[2], minimum[j])
            position[j] = point
    velocity = Mat43()
    for j in range(4):
        velocity[j] = (row4(position, j) - row4(old.position, j)) / DT
    for e in range(6):
        a, b = edge_a(e), edge_b(e)
        delta = row4(position, a) - row4(position, b)
        direction = delta / wp.max(EPS, reference_norm(delta))
        relative = wp.dot(row4(velocity, a) - row4(velocity, b), direction)
        impulse = direction * (relative * p.damping / (p.inv_mass[a] + p.inv_mass[b]))
        velocity[a] = row4(velocity, a) - p.inv_mass[a] * impulse
        velocity[b] = row4(velocity, b) + p.inv_mass[b] * impulse
    result = PayloadResult()
    next_state = PayloadState()
    for j in range(4):
        v = row4(velocity, j)
        normal_dv = float(0.)
        if position[j, 2] <= minimum[j]:
            normal_dv = wp.max(0., -v[2])
        total_dv = normal_shift[j] / DT + normal_dv
        friction = wp.min(1., p.friction * total_dv / wp.max(EPS, wp.length(wp.vec2(v[0], v[1]))))
        next_state.velocity[j] = wp.vec3(v[0] * (1. - friction), v[1] * (1. - friction), v[2] + normal_dv)
        result.support[j] = p.mass[j] * total_dv / DT
        result.clearance[j] = position[j, 2] - minimum[j]
    next_state.position = position
    next_state.depth = depth
    result.state = next_state
    return result


@wp.kernel
def advance_ropes(params: wp.array[RopeParameters], inputs: wp.array[RopeInput], states: wp.array[RopeState],
                  strict: bool, outputs: wp.array[RopeResult]):
    world = wp.tid()
    result = step_rope(params[world], states[world], inputs[world], strict)
    states[world] = result.state
    outputs[world] = result


@wp.kernel
def advance_blocks(params: wp.array[PayloadParameters], loads: wp.array[Mat43], depth: wp.array[wp.vec4],
                   states: wp.array[PayloadState], outputs: wp.array[PayloadResult]):
    world = wp.tid()
    result = advance_payload(params[world], states[world], loads[world], depth[world])
    states[world] = result.state
    outputs[world] = result


@wp.func
def material_weights(half: wp.vec3, local: wp.vec3):
    u = .5 * (1. + wp.clamp(2. * local[0] / half[0], -1., 1.))
    v = .5 * (1. + wp.clamp(2. * local[1] / half[1], -1., 1.))
    return wp.vec4((1. - u) * (1. - v), u * (1. - v), (1. - u) * v, u * v)


@wp.func
def weighted(weights: wp.vec4, values: Mat43):
    value = wp.vec3()
    for j in range(4):
        value += weights[j] * row4(values, j)
    return value


@wp.func
def normal_for(positions: Mat43):
    a = row4(positions, 3) - row4(positions, 0)
    b = row4(positions, 2) - row4(positions, 1)
    cross = wp.cross(a, b)
    if wp.length(cross) <= EPS:
        return wp.vec3(0., 0., 1.)
    return wp.normalize(cross)


@wp.func
def project_to_material(point: wp.vec3, positions: Mat43, normal: wp.vec3):
    x = row4(positions, 3) - row4(positions, 0) - row4(positions, 2) + row4(positions, 1)
    if wp.length(x) <= EPS:
        x = wp.vec3(1., 0., 0.)
    x = wp.normalize(x)
    relative = point - weighted(wp.vec4(.25), positions)
    return wp.vec3(wp.dot(relative, x), wp.dot(relative, wp.cross(normal, x)), wp.dot(relative, normal))


@wp.func
def jacobian_row(weights: wp.vec4, direction: wp.vec3, height: float, a: wp.vec3, b: wp.vec3,
                 normal: wp.vec3, area: float):
    adjoint = wp.vec3()
    if area > EPS:
        adjoint = height * (direction - wp.dot(direction, normal) * normal) / area
    ga = wp.cross(b, adjoint)
    gb = wp.cross(adjoint, a)
    result = Mat43()
    result[0] = weights[0] * direction - ga
    result[1] = weights[1] * direction - gb
    result[2] = weights[2] * direction + gb
    result[3] = weights[3] * direction + ga
    return result


@wp.func
def axial_mobility(a: Mat43, b: Mat43, inv_mass: wp.vec4):
    value = float(0.)
    for j in range(4):
        value += inv_mass[j] * wp.dot(row4(a, j), row4(b, j))
    return value


@wp.struct
class SurfaceResult:
    depth: wp.vec4
    pads: wp.vec2
    direct: wp.vec2
    blocks: wp.vec4


@wp.func
def surface_contact(p: PayloadParameters, old: PayloadState, tips: Mat23, speeds: Mat23,
                    weights: Mat24, demand: wp.vec2, engaged: wp.vec2i):
    predicted = Mat43()
    predicted_weights = Mat24()
    predicted_demand = wp.vec2()
    for i in range(4):
        point = row4(old.position, i) + DT * row4(old.velocity, i)
        point[2] = wp.max(point[2], p.height[i] - .5 * old.depth[i])
        predicted[i] = point
    normal = normal_for(predicted)
    for j in range(2):
        tip_end = row2(tips, j) + DT * row2(speeds, j)
        w = material_weights(p.half, project_to_material(tip_end, predicted, normal))
        predicted_weights[j] = w
        predicted_demand[j] = wp.dot(w, p.height) - wp.dot(tip_end - weighted(w, predicted), normal)
    result = SurfaceResult()
    for i in range(4):
        raw = wp.vec2(-1e10, -1e10)
        target = wp.vec2(-1e10, -1e10)
        for j in range(2):
            if engaged[j] != 0:
                if weights[j, i] > 0.:
                    raw[j] = demand[j] * weights[j, i]
                if predicted_weights[j, i] > 0.:
                    target[j] = predicted_demand[j] * predicted_weights[j, i]
        start = wp.max(raw[0], raw[1])
        end = wp.max(target[0], target[1])
        if start == -1e10:
            start = 0.
        if end == -1e10:
            end = 0.
        previous, limit = old.depth[i], p.max_depth[i]
        free = previous * p.rebound
        depth = wp.max(free, wp.min(end, limit))
        result.depth[i] = depth
        compress = float(0.)
        if end >= free and end > 0.:
            norm = wp.max(limit, EPS)
            ratio = (depth * depth + previous * previous) / (norm * norm)
            elastic = p.soft_k * .5 * (depth + previous) * (1. + .5 * p.hardening * ratio)
            rebound = wp.max(0., 1. - wp.max(0., previous - depth) / wp.max(EPS, previous - free))
            compress = elastic * rebound + p.compression * wp.max(0., (depth - previous) / DT)
        start_core, end_core = wp.max(0., start - limit), wp.max(0., end - limit)
        core = .5 * 10000. * (end_core + start_core)
        sweep = wp.abs(end - start)
        if sweep > EPS:
            core = .5 * 10000. * wp.abs(end_core * end_core - start_core * start_core) / sweep
        owners = float(raw[0] == start) + float(raw[1] == start)
        for j in range(2):
            if raw[j] == start:
                gain = weights[j, i] / wp.max(1., owners)
                result.pads[j] += gain * (compress + core)
                result.direct[j] += gain * core
    for i in range(4):
        result.blocks[i] = weights[0, i] * result.pads[0] + weights[1, i] * result.pads[1]
    return result


@wp.func
def surface_proximity(p: PayloadParameters, position: Mat43, point: wp.vec3, upper_normal: wp.vec3):
    # Material projection alone is insufficient when a tilted payload falls or
    # rotates. Bound a pad by the swept finite slab before evaluating pressure.
    lowest = float(10.)
    highest = float(-10.)
    for j in range(4):
        lowest = wp.min(lowest, position[j, 2] - p.height[j])
        highest = wp.max(highest, position[j, 2] + p.height[j])
    incline = wp.sqrt(wp.max(0., 1. - upper_normal[2] * upper_normal[2]))
    rim = .5 * wp.length(wp.vec2(p.half[0], p.half[1])) * incline
    return point[2] >= lowest - rim - .004 and point[2] <= highest + rim + .004


@wp.struct
class ContactResult:
    blocks: PayloadResult
    ropes: RopeResult
    reaction: Mat23
    direct: wp.vec2


@wp.func
def payload_contact(p: PayloadParameters, rp: RopeParameters, old: PayloadState, ropes: RopeState,
                    x: RopeInput, mounts: Mat24, strict: bool):
    normal = normal_for(old.position)
    a = row4(old.position, 3) - row4(old.position, 0)
    b = row4(old.position, 2) - row4(old.position, 1)
    area = wp.length(wp.cross(a, b))
    weights = Mat24()
    demands = wp.vec2()
    active = wp.vec2i()
    for j in range(2):
        tip = row2(x.tip, j)
        local = project_to_material(tip, old.position, normal)
        w = material_weights(p.half, local)
        weights[j] = w
        separation = wp.dot(tip - weighted(w, old.position), normal)
        demands[j] = wp.dot(w, p.height) - separation
        previous_gap = separation - DT * wp.dot(row2(x.tip_velocity, j), normal)
        active[j] = int(wp.abs(local[0]) <= p.half[0] and wp.abs(local[1]) <= p.half[1] and
                        surface_proximity(p, old.position, tip, normal) and
                        separation >= -wp.dot(w, p.height) - .008 and
                        (previous_gap >= 0. or wp.dot(w, old.depth) > EPS or tip[2] <= 0.))
    surface = surface_contact(p, old, x.tip, x.tip_velocity, weights, demands, active)
    normal_rate = wp.vec3()
    if area > EPS:
        da = row4(old.velocity, 3) - row4(old.velocity, 0)
        db = row4(old.velocity, 2) - row4(old.velocity, 1)
        derivative = wp.cross(da, b) + wp.cross(a, db)
        normal_rate = (derivative - wp.dot(derivative, normal) * normal) / area
    height = wp.vec2()
    directions = Mat23()
    indentation = wp.vec2()
    for j in range(2):
        w = wp.vec4(mounts[j, 0], mounts[j, 1], mounts[j, 2], mounts[j, 3])
        indentation[j] = wp.dot(w, surface.depth)
        height[j] = wp.dot(w, p.height) - .5 * indentation[j]
        x.anchor[j] = weighted(w, old.position) + height[j] * normal
        height_rate = -.5 * wp.dot(w, surface.depth - old.depth) / DT
        x.anchor_velocity[j] = weighted(w, old.velocity) + height[j] * normal_rate + height_rate * normal
        line = row2(x.tip, j) - row2(x.anchor, j)
        directions[j] = line / wp.max(EPS, wp.length(line))
        x.feed[j] = wp.min(x.feed[j], .10) + wp.min(indentation[j], .020)
        x.alignment[j] = wp.clamp(x.alignment[j] + indentation[j] / (indentation[j] + .005), 0., 1.)
    width = 2. * p.half[0]
    top_growth = wp.length(wp.vec2(width, indentation[0] - indentation[1])) - width
    total_indent = indentation[0] + indentation[1]
    x.budget += wp.min(.5 * total_indent, wp.max(0., total_indent - top_growth))
    w0 = wp.vec4(mounts[0, 0], mounts[0, 1], mounts[0, 2], mounts[0, 3])
    w1 = wp.vec4(mounts[1, 0], mounts[1, 1], mounts[1, 2], mounts[1, 3])
    ja = jacobian_row(w0, row2(directions, 0), height[0], a, b, normal, area)
    jb = jacobian_row(w1, row2(directions, 1), height[1], a, b, normal, area)
    cross_mobility = wp.abs(axial_mobility(ja, jb, p.inv_mass))
    x.mobility = wp.vec2(axial_mobility(ja, ja, p.inv_mass) + cross_mobility,
                         axial_mobility(jb, jb, p.inv_mass) + cross_mobility)
    result = ContactResult()
    result.ropes = step_rope(rp, ropes, x, strict)
    f0 = row2(result.ropes.force, 0)
    f1 = row2(result.ropes.force, 1)
    # Apply the transpose of anchor kinematics, including its surface-normal derivative.
    equivalent = height[0] * f0 + height[1] * f1
    adjoint = wp.vec3()
    if area > EPS:
        adjoint = (equivalent - wp.dot(equivalent, normal) * normal) / area
    ga, gb = wp.cross(b, adjoint), wp.cross(adjoint, a)
    load = Mat43()
    load[0] = -ga
    load[1] = -gb
    load[2] = gb
    load[3] = ga
    for j in range(4):
        load[j] = row4(load, j) + w0[j] * f0 + w1[j] * f1 - surface.blocks[j] * normal
    result.reaction[0] = surface.pads[0] * normal - f0
    result.reaction[1] = surface.pads[1] * normal - f1
    result.direct = surface.direct
    result.blocks = advance_payload(p, old, load, surface.depth)
    return result


@wp.kernel
def advance_suspended_reference(positions: wp.array3d[wp.vec3], p: wp.array[PayloadParameters],
                                rp: wp.array[RopeParameters], blocks: wp.array[PayloadState],
                                ropes: wp.array[RopeState], mounts: wp.array[Mat24], step: int,
                                samples: wp.array3d[float]):
    """Write a same-input suspended trajectory for the numerical C0 gate."""
    i = wp.tid()
    x = RopeInput()
    x.budget = rp[i].budget
    x.normal = wp.vec2(6.)
    for j in range(2):
        x.tip[j] = positions[i, step, j]
        x.tip_velocity[j] = (positions[i, step, j] - positions[i, step - 1, j]) / DT
    result = payload_contact(p[i], rp[i], blocks[i], ropes[i], x, mounts[i], False)
    blocks[i] = result.blocks.state
    ropes[i] = result.ropes.state
    samples[i, step - 1, 0] = result.ropes.tension[0]
    samples[i, step - 1, 1] = result.ropes.tension[1]
    samples[i, step - 1, 2] = result.ropes.state.contacts[0, 1]
    samples[i, step - 1, 3] = result.ropes.state.contacts[1, 1]
    clearance = result.blocks.clearance
    samples[i, step - 1, 4] = wp.min(wp.min(clearance[0], clearance[1]), wp.min(clearance[2], clearance[3]))
