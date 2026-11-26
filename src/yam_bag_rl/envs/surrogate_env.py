"""Vectorized low-fidelity Yam_B simulation with three host decision boundaries."""

from dataclasses import dataclass
from xml.etree import ElementTree as ET

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from ..bag.replay_config import MONDAY
from ..robot import yam_warp_kernels as motor_kernels
from ..robot.abc_workcell import add_workcell
from ..robot.kinematics import descendants
from ..robot.yam_model import initial_state
from mujoco_warp._src.types import OverflowType
from ..robot.yam_warp import SimYAMInterface
from . import surrogate_control as control
from . import surrogate_scene as parcel
from .surrogate_equivalence import rope_parameters

POLICY_PERIOD = .04
PHASE_TICKS = (8, 18, 13, 13, 25, 22, 17, 23, 65, 82, 23, 28)
HISTORY_FRAMES = sum(PHASE_TICKS[1:5])
ACTION_MASK = np.array([1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1], dtype=bool)
AUDIT_FIELDS = (
    "physics_validity", "invalid_reason", "probe_cutoff", "probe_contact", "released_tethers", "slipped_tethers",
    "active_tethers", "peak_clearance_m", "final_clearance_m", "final_speed_m_s", "robot_payload_direct_force",
    "robot_payload_direct_impulse_Ns", "peak_tension_N", "normalized_effort_cost", "peak_tau_res_Nm",
    "lift_instability", "support_N", "rope_vertical_N", "peak_solver_iterations", "retracted", "task_success",
    "overflow_flags", "peak_fold_N", "peak_feed_m", "peak_gather_m", "probe_unloaded", "rigid_collision",
)
wp.set_module_options({"enable_backward": False})


@wp.kernel
def pack_try(telemetry: control.Telemetry, candidates: wp.array3d[float], out: wp.array2d[float]):
    env, j = wp.tid()
    w = telemetry.history.shape[2] + 3
    length = telemetry.history.shape[1] * w
    if j < length:
        frame, k = j // w, j % w
        if k < w - 3:
            out[env, j] = telemetry.history[env, frame, k]
        elif k == w - 3:
            out[env, j] = float(telemetry.valid[env, frame])
        elif k == w - 2:
            out[env, j] = float(telemetry.windows[env, frame])
        else:
            out[env, j] = telemetry.stamps[env, frame]
    else:
        k = j - length
        out[env, j] = candidates[env, k // 11, k % 11]


@wp.kernel
def pack_end(scene: parcel.Scene, out: wp.array2d[float]):
    i = wp.tid()
    audit = scene.audits[i]
    out[i, 0] = float(audit.invalid == 0)
    out[i, 1] = float(audit.invalid)
    out[i, 2] = float(audit.cutoff)
    out[i, 3] = float(audit.probe_contact)
    out[i, 4] = float(audit.released)
    out[i, 5] = float(audit.slipped)
    out[i, 6] = float(audit.captured)
    out[i, 7] = audit.peak_clearance
    out[i, 8] = audit.current_clearance
    out[i, 9] = audit.last_speed
    out[i, 10] = audit.direct_peak
    out[i, 11] = audit.direct_integral
    out[i, 12] = audit.peak_tension
    out[i, 13] = audit.probe_effort_cost / float(wp.max(1, audit.probe_samples))
    out[i, 14] = audit.peak_residual
    out[i, 15] = wp.clamp(audit.transport_speed / .7, 0., 1.)
    out[i, 16] = audit.support_N
    out[i, 17] = audit.rope_up_N
    out[i, 18] = float(audit.peak_solver_iterations)
    out[i, 19] = float(audit.retracted)
    out[i, 20] = float(audit.success != 0 and audit.invalid == 0 and audit.cutoff != 2 and
                       scene.collisions[i] == 0 and audit.retracted != 0)
    out[i, 21] = float(audit.overflow_flags)
    out[i, 22] = audit.peak_fold_N
    out[i, 23] = audit.peak_feed_m
    out[i, 24] = audit.peak_gather_m
    out[i, 25] = float(audit.probe_unloaded)
    out[i, 26] = float(scene.collisions[i])
    for j in range(14):
        out[i, 27 + j] = scene.hidden[i, j]
    out[i, 41] = float(scene.regime[i])


@wp.kernel
def select_commands(actions: wp.array2d[float], mask: wp.array[bool], out: wp.array2d[float]):
    env, j = wp.tid()
    out[env, j] = wp.clamp(actions[env, j], -1., 1.) * float(mask[j])


@wp.kernel
def trace_frame(scene: parcel.Scene, qpos: wp.array2d[float], site: wp.array2d[wp.vec3],
                site_id: int, t: int, traces: wp.array3d[float]):
    i = wp.tid()
    st = scene.state[i]
    audit = scene.audits[i]
    for j in range(qpos.shape[1]):
        traces[i, t, j] = qpos[i, j]
    for j in range(4):
        for k in range(3):
            traces[i, t, 8 + 3 * j + k] = st.blocks.position[j, k]
        traces[i, t, 20 + j] = st.payload.height[j] - .5 * st.blocks.depth[j]
    for k in range(3):
        traces[i, t, 24 + k] = site[i, site_id][k]
    traces[i, t, 27] = float(scene.stage[i])
    traces[i, t, 28] = audit.current_clearance
    traces[i, t, 29] = float(audit.captured)
    traces[i, t, 30] = audit.direct_now
    traces[i, t, 31] = audit.support_N
    traces[i, t, 32] = audit.rope_up_N
    traces[i, t, 33] = float(audit.cutoff)
    traces[i, t, 34] = audit.peak_tension


@dataclass(frozen=True)
class TryBatch:
    history: np.ndarray
    valid: np.ndarray
    windows: np.ndarray
    timestamps: np.ndarray
    candidates: np.ndarray


@dataclass(frozen=True)
class EpisodeBatch:
    audit: dict
    private: np.ndarray
    regime: np.ndarray

    @property
    def valid(self):
        return self.audit["physics_validity"].astype(bool)

    @property
    def success(self):
        return self.audit["task_success"].astype(bool)


def _compile_yam(xml):
    root = ET.fromstring(xml)
    option = root.find("option")
    option.set("iterations", "12")
    option.set("ls_iterations", "9")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def _check_equalities(model):
    # The stock Yam model has one joint-equality coupling the left and right rails.
    # Reject all other equalities before allocating any worlds.
    expected = {model.joint(name).id for name in ("left_finger", "right_finger")}
    if (model.neq != 1 or int(model.eq_type[0]) != int(mujoco.mjtEq.mjEQ_JOINT) or
            {int(model.eq_obj1id[0]), int(model.eq_obj2id[0])} != expected):
        raise RuntimeError("Structural physics failure: unexpected equality in Yam_B scene")


class VectorSurrogate:
    """MJWarp robot coupled by device-side generalized forces to the Monday proxy."""

    def __init__(self, worlds, *, device="cuda:0", candidates=16, trace=False, action_mask=ACTION_MASK):
        self.worlds = worlds
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError("The surrogate requires a CUDA Warp device")
        if candidates < 12 or candidates > 256:
            raise ValueError("Use 12–256 continuously perturbed grasp candidates")
        self.count = candidates
        self.record_trace = trace
        self.graph = None
        self.base_id = 0
        self.seed = 0
        self.sensor_seed = 5291
        self.phase_position = 0
        overrides = dict(physics_dt_s=.002, actuator_hz=500, observation_hz=500, command_hz=500)
        self.yam = SimYAMInterface(worlds, device=device, config=overrides, add_scene=add_workcell,
                                   compile_model=_compile_yam, capture=False,
                                   capacities=dict(nconmax=40, nccdmax=24, njmax=160))
        self.model = self.yam.model
        self.yam.gpu_model.opt.warn_overflow &= ~int(OverflowType.ITERATIONS | OverflowType.LS_ITERATIONS)
        _check_equalities(self.model)
        self.dt = self.yam.dt
        robot = self.yam.robot
        home = initial_state(self.model, self.yam.config)
        site = self.model.site("grasp_site").id
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, home.site_xmat[site])
        with wp.ScopedDevice(self.device):
            self.scene = parcel.Scene()
            self.scene.state = wp.zeros(worlds, dtype=parcel.World)
            self.scene.audits = wp.zeros(worlds, dtype=parcel.Audit)
            self.scene.regime = wp.zeros(worlds, dtype=int)
            self.scene.v0 = wp.zeros((worlds, 12), dtype=float)
            self.scene.hidden = wp.zeros((worlds, 14), dtype=float)
            self.scene.stage = wp.zeros(worlds, dtype=int)
            self.scene.collisions = wp.zeros(worlds, dtype=int)
            groups = np.zeros(self.model.ngeom, dtype=np.int32)
            arm = np.isin(self.model.geom_bodyid, list(descendants(self.model, "arm")))
            groups[arm] = 1
            groups[self.model.geom_bodyid == self.model.body("arm").id] = 2
            groups[self.model.geom("table").id] = 3
            self.collision_groups = wp.array(groups, dtype=int)
            self.scene.pad_bodies = wp.array([self.model.body(name).id for name in ("left_pad", "right_pad")], dtype=int)
            self.scene.pad_offsets = wp.array([[0, 0, -.1444]] * 2, dtype=wp.vec3)
            self.scene.jaw_effort = wp.zeros(worlds, dtype=float)
            self.motion = control.Motion()
            self.motion.actions = wp.zeros((worlds, 13), dtype=float)
            self.raw_actions = wp.zeros_like(self.motion.actions)
            self.motion.candidates = wp.zeros((worlds, candidates, 11), dtype=float)
            self.motion.selected = wp.zeros(worlds, dtype=int)
            self.motion.target_position = wp.zeros(worlds, dtype=wp.vec3)
            self.motion.target_rotation = wp.zeros(worlds, dtype=wp.quat)
            self.motion.unload = wp.zeros((worlds, 7), dtype=float)
            self.motion.hover_q = wp.array(self.yam.config["ready_joints_rad"], dtype=float)
            self.motion.hover_xyz = wp.vec3(*home.site_xpos[site])
            self.motion.hover_quat = wp.quat(q[1], q[2], q[3], q[0])
            self.motion.lo = wp.vec3(-.18, -.34, .012)
            self.motion.hi = wp.vec3(.09, .035, .32)
            self.motion.skip_try = wp.zeros(worlds, dtype=int)
            self.motion.fault = wp.zeros(worlds, dtype=int)
            self.mask = wp.array(np.asarray(action_mask, dtype=bool), dtype=bool)
            self.telemetry = t = control.Telemetry()
            for name in ("filters", "gain", "bias", "noise", "alpha", "deadband"):
                setattr(t, name, wp.zeros((worlds, 7), dtype=float))
            t.dropout = wp.zeros((worlds, 7), dtype=int)
            for name in ("sensor_delay", "control_delay", "counter", "identity"):
                setattr(t, name, wp.zeros(worlds, dtype=int))
            t.commands = wp.zeros((worlds, 16, 7), dtype=float)
            t.ring = wp.zeros((worlds, 16, control.SENSOR_FIELDS), dtype=float)
            t.history = wp.zeros((worlds, HISTORY_FRAMES, control.SENSOR_FIELDS), dtype=float)
            t.valid = wp.zeros((worlds, HISTORY_FRAMES), dtype=bool)
            t.windows = wp.zeros((worlds, HISTORY_FRAMES), dtype=int)
            t.stamps = wp.zeros((worlds, HISTORY_FRAMES), dtype=float)
            t.nominal_mass = wp.array(self.model.body_mass, dtype=float)
            bodies = []
            for j in range(6):
                sub = sorted(descendants(self.model, f"link_{j + 1}"))
                bodies.append(sub + [-1] * (self.model.nbody - len(sub)))
            t.descendants = wp.array(bodies, dtype=int)
            self.actual_cmd = motor_kernels.Commands()
            for field in ("requested", "velocity", "desired_velocity", "feedforward", "action"):
                setattr(self.actual_cmd, field, getattr(self.yam.cmds, field))
            self.actual_cmd.target = wp.zeros((worlds, 7), dtype=float)
            self.packed_try = wp.zeros((worlds, HISTORY_FRAMES * (control.SENSOR_FIELDS + 3) + candidates * 11),
                                       dtype=float)
            self.packed_end = wp.zeros((worlds, len(AUDIT_FIELDS) + 15), dtype=float)
            if trace:
                self.trace = wp.zeros((worlds, sum(PHASE_TICKS), 35), dtype=float)
        self.original = rope_parameters(MONDAY["proxy"])
        self.basket = wp.vec3(-.125, -.045, .0)
        self._reset_device(0, 0, False, True)
        self._prepare(0)
        self._advance_microstep()
        with wp.ScopedDevice(self.device):
            wp.synchronize_device(self.device)
            with wp.ScopedCapture() as cap:
                for _ in range(20):
                    self._advance_microstep()
            self.graph = cap.graph

    def _launch(self, kernel, shape, *params):
        wp.launch(kernel, shape, params, device=self.device)

    def _prepare(self, phase, corridor=False):
        self._launch(control.prepare_phase, self.worlds, self.scene, self.motion, phase, self.basket, corridor)

    def _advance_microstep(self):
        robot, cmd, motors = self.yam.robot, self.yam.cmds, self.yam.motors
        m, d = self.yam.gpu_model, self.yam.data
        mjw.step1(m, d)
        self._launch(control.stage_command, self.worlds, self.scene, self.motion, motors, robot, cmd, d.qpos,
                     d.site_xpos, d.site_xmat, d.xanchor, d.xaxis)
        self._launch(motor_kernels.slew_targets, (self.worlds, 7), robot, cmd, self.dt)
        self._launch(control.command_delay, self.worlds, self.telemetry, cmd, self.actual_cmd.target)
        self._launch(motor_kernels.motor_control, (self.worlds, 7), robot, self.actual_cmd, motors, self.dt,
                     d.qpos, d.qvel, d.ctrl, m.body_mass, d.xipos, d.xanchor, d.xaxis)
        self._launch(parcel.couple_parcel, self.worlds, self.scene, robot, cmd, motors, d.qpos, d.xpos, d.xmat,
                     d.xanchor, d.xaxis, d.qfrc_applied)
        mjw.step2(m, d)
        self._launch(control.audit_collision_depth, d.naconmax, d.nacon, d.contact.worldid,
                     d.contact.geom, d.contact.dist, self.collision_groups, self.scene.collisions)
        self._launch(control.sense_and_guard, self.worlds, self.scene, self.motion, self.telemetry, motors, robot,
                     self.actual_cmd, d.qpos, d.qvel, d.xipos, d.xanchor, d.xaxis, d.site_xpos, d.site_xmat,
                     d.actuator_force, d.overflow, d.solver_niter, d.time, self.sensor_seed)

    def _reset_device(self, seed, base_id, training, fixed_regimes):
        self.yam.reset()
        self._launch(parcel.reset_parcel, self.worlds, self.scene, self.original, seed, base_id, int(fixed_regimes), 0)
        self._launch(control.reset_telemetry, self.worlds, self.telemetry, seed + 293, base_id, training)
        self.motion.skip_try.zero_()
        self.motion.fault.zero_()
        self.scene.collisions.zero_()
        self.phase_position = 0

    def _set(self, destination, values, dtype):
        values = np.asarray(values, dtype=dtype)
        if destination.shape != values.shape or not np.isfinite(values).all():
            raise ValueError(f"Expected finite shape {destination.shape}, got {values.shape}")
        wp.copy(destination, wp.array(values, dtype=destination.dtype, device=self.device))

    def _phase(self, phase, frames=None):
        self._prepare(phase)
        for tick in range(PHASE_TICKS[phase]):
            if phase == control.LIFT and tick == 32:
                self._prepare(phase, corridor=True)
            wp.capture_launch(self.graph)
            if frames is not None:
                self._launch(control.record_history, self.worlds, self.scene, self.motion, self.telemetry,
                             frames + tick, self.dt)
            if self.record_trace:
                self._launch(trace_frame, self.worlds, self.scene, self.yam.data.qpos, self.yam.data.site_xpos,
                             self.yam.robot.site, self.phase_position, self.trace)
            self.phase_position += 1

    def reset(self, seed, base_id, *, training=False, fixed_regimes=True):
        """Start a fresh batch and return the public V0 vector for the actor."""
        self.seed = seed
        self.base_id = base_id
        with wp.ScopedDevice(self.device):
            self._reset_device(seed, base_id, training, fixed_regimes)
            self._phase(0)
        return self.scene.v0.numpy()

    def try_batch(self, actions, *, skip=None, collection=False, override_candidates=None):
        """Advance a whole Try including the return to q_hover, then copy scorer inputs."""
        with wp.ScopedDevice(self.device):
            self._set(self.raw_actions, actions, np.float32)
            self._launch(select_commands, self.motion.actions.shape, self.raw_actions, self.mask, self.motion.actions)
            if skip is not None:
                self._set(self.motion.skip_try, skip, np.int32)
            frame = 0
            for phase in range(1, control.RETRACT + 1):
                self._phase(phase, frame)
                frame += PHASE_TICKS[phase]
            if override_candidates is None:
                self._launch(control.sample_candidates, (self.worlds, self.count), self.scene.v0,
                             self.seed + 1271, self.telemetry.identity, self.motion.hover_quat, int(collection),
                             self.motion.candidates)
            else:
                self._set(self.motion.candidates, override_candidates, np.float32)
            self._launch(pack_try, self.packed_try.shape, self.telemetry, self.motion.candidates, self.packed_try)
        batch = self.packed_try.numpy()
        split = HISTORY_FRAMES * (control.SENSOR_FIELDS + 3)
        frames = batch[:, :split].reshape(self.worlds, HISTORY_FRAMES, control.SENSOR_FIELDS + 3)
        return TryBatch(frames[:, :, :-3], frames[:, :, -3].astype(bool), frames[:, :, -2].astype(np.int32),
                        frames[:, :, -1], batch[:, split:].reshape(self.worlds, self.count, 11))

    def grasp_batch(self, indices):
        """Execute the selected public SE(3) grasps and download macro outcomes."""
        indices = np.asarray(indices)
        if not np.issubdtype(indices.dtype, np.integer) or np.any((indices < 0) | (indices >= self.count)):
            raise ValueError("Candidate selections must be valid integer indices")
        with wp.ScopedDevice(self.device):
            self._set(self.motion.selected, indices, np.int32)
            for stage in range(control.GRASP_PRE, control.SETTLE + 1):
                self._phase(stage)
            self._launch(pack_end, self.worlds, self.scene, self.packed_end)
        result = self.packed_end.numpy()
        audit = {name: result[:, i] for i, name in enumerate(AUDIT_FIELDS)}
        return EpisodeBatch(audit, result[:, len(AUDIT_FIELDS):-1], result[:, -1].astype(np.int32))

    def set_c0_faults(self, modes):
        if not self.record_trace:
            raise RuntimeError("Fault injection is restricted to traced C0 tests")
        with wp.ScopedDevice(self.device):
            self._set(self.motion.fault, modes, np.int32)
            self._launch(control.configure_faults, self.worlds, self.scene, self.motion.fault)

    def _integration_state(self):
        d = self.yam.data
        t = self.telemetry
        cmd = self.yam.cmds
        motors = self.yam.motors
        return (d.qpos, d.qvel, d.act, d.time, d.qacc_warmstart, d.ctrl, d.qfrc_applied,
                self.scene.state, self.scene.audits, self.scene.stage, self.scene.jaw_effort, self.scene.collisions,
                self.motion.skip_try, self.motion.unload, self.motion.target_position, self.motion.target_rotation,
                t.filters, t.ring, t.counter, t.commands, cmd.requested, cmd.target, self.actual_cmd.target,
                motors.command, motors.applied, motors.tuning)

    def snapshot_try(self):
        """Branch paired grasps from the same post-Try integration and telemetry state."""
        if self.phase_position != sum(PHASE_TICKS[:5]):
            raise RuntimeError("A grasp snapshot requires a completed Try")
        return self.phase_position, tuple(wp.clone(a) for a in self._integration_state())

    def restore_try(self, snapshot):
        frame, state = snapshot
        for target, source in zip(self._integration_state(), state, strict=True):
            wp.copy(target, source)
        self.phase_position = frame
