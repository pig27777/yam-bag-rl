"""Native 25 Hz Yam_B control with the shared motor calibration."""

import mujoco
import numpy as np

from ..policy_v3.environment import MotionStop
from .kinematics import descendants
from .yam_model import arm_indices, initial_state


class CPURigidDrive:
    """Native MuJoCo rigid-only smoke driver with the Yam_B nominal motor contract."""

    name = "mujoco_cpu"

    def __init__(self, model, cfg, motor_Nm_per_N):
        self.model = model
        self.data = initial_state(model, cfg)
        self.initial = self.data.qpos.copy()
        joints, self.qadr, self.dofs = arm_indices(model)
        self.gear = motor_Nm_per_N
        self.preload = 2 * model.jnt_range[joints[6], 0]
        self.kp = np.array([*cfg["kp"], cfg["gripper_kp_N_m"]])
        self.kd = np.array([*cfg["kd"], cfg["gripper_kd_N_s_m"]])
        self.caps = np.array([*cfg["torque_limit_Nm"], cfg["gripper_force_limit_N"]], dtype=float)
        self.speed = np.array([*cfg["joint_target_speed_rad_s"], cfg["gripper_speed_m_s"]])
        self.gravity_factor = np.array(cfg["gravity_factor"])
        self.max_gap = cfg["gripper_max_gap_m"]
        self.dt = cfg["physics_dt_s"]
        self.motor_steps = round(1. / (self.dt * cfg["actuator_hz"]))
        self.command_steps = round(1. / (self.dt * cfg["command_hz"]))
        text_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXT, "gravity_bodies")
        gravity_text = model.text_data[model.text_adr[text_id]:].split(b"\x00", 1)[0].decode()
        gravity_ids = [model.body(name).id for name in gravity_text.split()]
        mask = np.array([[body in descendants(model, model.body(model.jnt_bodyid[j]).name) for body in gravity_ids]
                         for j in joints[:6]])
        self.links = np.array(gravity_ids)
        self.weights = mask * model.body_mass[self.links][None]
        self.joints = joints[:6]

    def reset(self, q):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = q
        self.target = np.r_[q[self.qadr[:6]], q[self.qadr[6:]].sum()]
        self.command = np.r_[np.zeros(6), 2 * self.target[-1] / self.max_gap - 1]
        mujoco.mj_forward(self.model, self.data)
        self.stopped = False
        self._read()

    def _read(self):
        d = self.data
        self.timestamp = float(d.time)
        self.observation = np.r_[d.qpos[self.qadr[:6]], d.qvel[self.dofs[:6]], d.actuator_force[:6], self.target[:6],
                                 d.qpos[self.qadr[6:]].sum(), d.qvel[self.dofs[6:]].sum(),
                                 d.actuator_force[6] * self.gear, self.target[-1]]

    def stop(self):
        self.stopped = True
        self.data.ctrl[:] = 0.

    def tick(self, command, effort, check, phase):
        if self.stopped:
            raise RuntimeError("Drive is stopped until reset")
        m, d = self.model, self.data
        self.command = np.r_[np.clip((command[:6] - self.target[:6]) / (.04 * self.speed[:6]), -1, 1),
                             2 * command[-1] / self.max_gap - 1]
        caps = self.caps.copy()
        caps[-1] = effort / self.gear
        slew = self.dt * self.command_steps * self.speed
        start = d.time
        for i in range(round(.04 / self.dt)):
            if i % self.command_steps == 0:
                self.target += np.clip(command - self.target, -slew, slew)
            mujoco.mj_step1(m, d)
            if i % self.motor_steps == 0:
                q = np.r_[d.qpos[self.qadr[:6]], d.qpos[self.qadr[6:]].sum()]
                dq = np.r_[d.qvel[self.dofs[:6]], d.qvel[self.dofs[6:]].sum()]
                target = self.target.copy()
                target[-1] += self.preload * (1 - target[-1] / self.max_gap)
                levers = d.xipos[self.links][None] - d.xanchor[self.joints, None]
                forces = self.weights[..., None] * m.opt.gravity
                gravity = -np.einsum("ij,ij->i", np.cross(levers, forces).sum(axis=1), d.xaxis[self.joints])
                torque = self.kp * (target - q) - self.kd * dq
                torque[:6] += gravity * self.gravity_factor
                d.ctrl[:] = np.clip(torque, -caps, caps)
            mujoco.mj_step2(m, d)
            if (d.time <= start or any(d.warning[index].number for index in (
                    mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
                    mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADCTRL,
                    mujoco.mjtWarning.mjWARN_CNSTRFULL, mujoco.mjtWarning.mjWARN_CONTACTFULL))):
                raise MotionStop("solver_warning", physics_valid=False)
            if d.ncon:
                check()
        self._read()
        if not np.isfinite(self.observation).all():
            raise MotionStop("nonfinite_physics", physics_valid=False)
