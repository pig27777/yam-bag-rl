"""Compile the untouched robot geometry and operate the local ABC controller."""

import unittest

import mujoco
import numpy as np

from yam_bag_rl.envs import make_smoke_env
from yam_bag_rl.robot.abc_workcell import ABC, add_workcell
from yam_bag_rl.robot.yam_model import ASSET, arm_indices, build_yam_model, initial_state, load_config, nominal_gravity_bias


class YamAssetsTest(unittest.TestCase):
    def test_nominal_mjcf_and_abc_workcell(self):
        cfg = load_config("nominal")
        raw = mujoco.MjModel.from_xml_path(str(ASSET / "yam_linear_4310.xml"))
        self.assertEqual(raw.nmesh, 15)
        _, workcell = build_yam_model(add_scene=add_workcell)
        data = initial_state(workcell, cfg)
        np.testing.assert_allclose(data.body("arm").xpos, ABC["arm_bases_m"][1], rtol=0., atol=1e-12)
        for name in ("abc_top", "abc_wrist_b"):
            self.assertGreaterEqual(workcell.camera(name).id, 0)
        self.assertTrue(np.isfinite(data.site("grasp_site").xpos).all())
        _, qadr, dofs = arm_indices(raw)
        for offset in (0., .1):
            pose = np.array(cfg["ready_joints_rad"])
            pose[1] += offset
            direct = mujoco.MjData(raw)
            direct.qpos[qadr[:6]] = pose
            direct.qpos[qadr[6:]] = cfg["ready_gap_m"] / 2
            self.assertTrue((direct.qvel == 0).all())
            mujoco.mj_forward(raw, direct)
            bias = nominal_gravity_bias(pose)
            self.assertTrue(np.isfinite(bias).all())
            self.assertGreater(np.linalg.norm(bias), 1.)
            np.testing.assert_allclose(bias, direct.qfrc_bias[dofs[:6]], rtol=0., atol=1e-12)
        print(f"{raw.nmesh} mesh assets, workcell FK {data.site('grasp_site').xpos}, bias {bias} N·m", flush=True)

    def test_control_step_and_macro_interface(self):
        env = make_smoke_env()
        try:
            v0, _, _ = env.reset()
            backend = env.backend
            before = backend.drive.timestamp
            backend.tick(backend.last_command.copy(), backend.probe_effort_Nm, "probe")
            self.assertAlmostEqual(backend.drive.timestamp - before, .04, places=10)
            self.assertTrue(np.isfinite(backend.drive.observation).all())
            self.assertTrue(np.isfinite(v0.features()).all())
            self.assertEqual(v0.features().shape, (12,))
        finally:
            env.close()
