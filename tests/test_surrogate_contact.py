"""A dropped, inverted payload must not exert pressure on faraway pads."""

import numpy as np
import warp as wp

from yam_bag_rl.bag.replay_config import MONDAY
from yam_bag_rl.envs import surrogate_kernels as gpu
from yam_bag_rl.envs.surrogate_equivalence import rope_parameters

wp.set_module_options({"enable_backward": False, "fuse_fp": False})


@wp.kernel
def scene_contact(params: gpu.PayloadParameters, rope: gpu.RopeParameters, direct: wp.array[float]):
    i = wp.tid()
    state = gpu.PayloadState()
    longitudinal = float(1.)
    height = float(.052)
    if i == 0:
        longitudinal = -1.
        height = .295
    state.position[0] = wp.vec3(-.04, -.03 * longitudinal, .016)
    state.position[1] = wp.vec3(.04, -.03 * longitudinal, .016)
    state.position[2] = wp.vec3(-.04, .03 * longitudinal, .016)
    state.position[3] = wp.vec3(.04, .03 * longitudinal, .016)
    state.depth = wp.vec4(.002)
    x = gpu.RopeInput()
    x.tip[0] = wp.vec3(-.005, .002, height)
    x.tip[1] = wp.vec3(.005, .002, height)
    if i == 1:
        x.tip[0] = wp.vec3(-.005, .002, .020)
        x.tip[1] = wp.vec3(.005, .002, .020)
    mounts = gpu.Mat24()
    mounts[0] = wp.vec4(.25)
    mounts[1] = wp.vec4(.25)
    result = gpu.payload_contact(params, rope, state, gpu.RopeState(), x, mounts, True)
    direct[i] = result.direct[0] + result.direct[1]


def test_finite_core_contact_slab():
    p = gpu.PayloadParameters()
    p.half = wp.vec3(.08, .06, .016)
    p.height = wp.vec4(.016)
    p.mass = wp.vec4(.05)
    p.inv_mass = wp.vec4(20.)
    p.rest = gpu.Mat43(-.04, -.03, 0., .04, -.03, 0., -.04, .03, 0., .04, .03, 0.)
    p.stretch_alpha = 30.
    p.bend_alpha = 80.
    p.bend_limit = .001
    p.rebound = .95
    p.max_depth = wp.vec4(.001)
    direct = wp.zeros(2, dtype=float, device="cpu")
    wp.launch(scene_contact, 2, [p, rope_parameters(MONDAY["proxy"]), direct], device="cpu")
    result = direct.numpy()
    np.testing.assert_equal(result[0], 0.)
    assert result[1] > .2
