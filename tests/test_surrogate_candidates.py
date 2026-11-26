"""Candidate pools are fully determined by public V0 and their own seed."""

import numpy as np
import warp as wp

from yam_bag_rl.bag.replay_config import MONDAY
from yam_bag_rl.envs.surrogate_control import sample_candidates
from yam_bag_rl.envs.surrogate_equivalence import rope_parameters
from yam_bag_rl.envs.surrogate_scene import Audit, Scene, World, reset_parcel


def test_same_public_geometry_and_seed_reproduce_pool_across_hidden_states():
    device = "cpu"
    n, k = 8, 16
    scene = Scene()
    scene.state = wp.zeros(n, dtype=World, device=device)
    scene.audits = wp.zeros(n, dtype=Audit, device=device)
    scene.regime = wp.zeros(n, dtype=int, device=device)
    scene.v0 = wp.zeros((n, 12), dtype=float, device=device)
    scene.hidden = wp.zeros((n, 14), dtype=float, device=device)
    scene.stage = wp.zeros(n, dtype=int, device=device)
    scene.jaw_effort = wp.zeros(n, dtype=float, device=device)
    ident = wp.array(np.arange(n, dtype=np.int32), dtype=int, device=device)
    out = wp.zeros((n, k, 11), dtype=float, device=device)
    hover = wp.quat(1., 0., 0., 0.)
    rp = rope_parameters(MONDAY["proxy"])

    def sample(hidden_offset):
        wp.launch(reset_parcel, n, [scene, rp, 33, 0, 1, hidden_offset], device=device)
        wp.launch(sample_candidates, (n, k), [scene.v0, 771, ident, hover, 1, out], device=device)
        return scene.v0.numpy().copy(), scene.hidden.numpy().copy(), out.numpy().copy()

    public, hidden, candidates = sample(0)
    same_public, different_hidden, same_candidates = sample(919)
    np.testing.assert_array_equal(public, same_public)
    np.testing.assert_array_equal(candidates, same_candidates)
    assert np.any(different_hidden != hidden)
