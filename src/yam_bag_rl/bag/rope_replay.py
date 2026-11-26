"""Reproduce the 65 g Monday grasp from an eight-axis Yam_B joint trace."""
import argparse
import time
from pathlib import Path
import mujoco as mj
import numpy as np
import torch
from ..robot.yam_model import arm_indices, build_yam_model
from .lumped_contact_proxy import LumpedContactProxy, ProxyParams, ContactState
from .macro_parcel_sim import MacroParcelSim, ParcelAction
from .neck_damping import NeckDampingParams
from .pad_torsion import PadTorsionParams
from .unified_payload import PayloadParamsX, UnifiedPayloadTemplate
from .replay_config import MONDAY, PAYLOAD

CASE = MONDAY["neck_98mm"]

@torch.inference_mode()
def run(source, out, *, proxy_type=LumpedContactProxy):
    dt = MONDAY["step_s"]
    mass = CASE["mass_kg"]
    dims = np.asarray(CASE["cloth_size_m"])
    envelope = dict(MONDAY["proxy"], L0=[CASE["neck_length_m"]] * 2)
    fill_volume = np.prod([*MONDAY["flat_bag_xy_m"], dims[2]])
    params = PayloadParamsX.from_presets(
        PAYLOAD, ['folded_clothes'], bag_half_extents=torch.tensor(dims[None] / 2, dtype=torch.float32),
        total_mass=torch.tensor([[mass]]),
        fill_ratio=torch.tensor([[float(np.prod(dims) / fill_volume)]])
    )
    payload = UnifiedPayloadTemplate(params)
    radius, neck_length = CASE["grip_radius_m"], CASE["neck_length_m"]
    torsion = PadTorsionParams(radius_m=radius, neck_length_m=neck_length,
                              stiffness_Nm_rad=mass * 9.81 * radius ** 2 / neck_length)
    sim = MacroParcelSim(proxy_type(ProxyParams.from_scalars(envelope), dt), payload=payload,
                         neck_damping=NeckDampingParams(MONDAY["neck_damping"]), torsion=torsion)
    spacing = CASE["mount_half_spacing_m"]
    sim.payload_mount_weights = payload.weights_at(torch.tensor([[[-spacing, 0.], [spacing, 0.]]]))
    state=sim.initial_state()
    shift=state.com.new_tensor([*MONDAY["center_xy_m"], 0])
    state.payload.pos.add_(shift); state.com.add_(shift)
    zeros=torch.zeros(1,2)
    keys='time centers frame half_height capture anchors reference clearance support velocity com tension normal pads'
    frames={k:[] for k in keys.split()}
    summary=[]
    with np.load(source) as trace:
        q = trace["q_path"]
    _, robot = build_yam_model()
    kin = mj.MjData(robot)
    _, ids, _ = arm_indices(robot)

    def fk(joint_pos):
        kin.qpos[ids] = joint_pos
        mj.mj_kinematics(robot, kin)
        faces = []
        for name in ('left_pad', 'right_pad'):
            body = kin.body(name)
            faces.append(body.xpos + body.xmat.reshape(3, 3) @ np.array([0., 0., -.1444]))
        return np.stack(faces)

    prev=fk(q[0])
    start=time.perf_counter()
    for i,angles in enumerate(q):
        pad=fk(angles)
        vel=(pad-prev)/dt
        prev=pad
        axis=pad[1]-pad[0];gap=np.linalg.norm(axis);axis/=gap
        inward=-((vel-vel.mean(0))*np.stack([-axis,axis])).sum(1)
        tip=state.com.new_tensor(pad)[None]
        b=state.payload;R=payload.frame(b)
        local=torch.bmm(tip-b.pos.mean(1,keepdim=True),R)[0]
        in_film=(local[:,:2].abs()<=params.bag_half_extents[0,:2]).all(1)
        touch=in_film & (local[:,2]<=payload.half_height[0,0]) & (local[:,2]>-payload.half_height[0,0])
        captured=state.contacts[0,:,ContactState.CAPTURE]>.5
        normal=(touch|captured)*MONDAY["fold_normal_stiffness_N_m"]*max(0.,(MONDAY["fold_width_m"]-gap)/2)
        if i%20==0:
            anchors,_=payload.anchors(b,sim.payload_mount_weights,torch.zeros_like(b.d_surf))
            for k,v in dict(time=i*dt,centers=b.pos[0],frame=R[0],half_height=payload.half_thickness(b.d_surf)[0],
                    capture=captured,anchors=anchors[0],reference=state.contact_reference[0],pads=pad,normal=normal,
                    com=state.com[0],velocity=state.velocity[0],clearance=float(result.table_clearance) if i else 0.,
                    support=float(result.support_N) if i else 0.,tension=result.tension[0] if i else zeros[0]).items():
                frames[k].append(np.asarray(v).copy())
        if i==len(q)-1:break
        vel_tensor=tip.new_tensor(vel)[None]
        feed = torch.where(touch, payload.half_height - local[:, 2], zeros).clamp_min(0)
        normal_component = tip.new_tensor(axis) @ R[0, :, 2]
        alignment = torch.where(touch, (1 - normal_component.square()).clamp_min(0).sqrt(), zeros)
        action=ParcelAction(tip,vel_tensor,normal[None],tip.new_tensor(inward)[None],feed,alignment,
                            torch.zeros(1,3),torch.zeros(1,1))
        wrist=kin.body('link_6').xquat.copy()
        angular=np.zeros(3)
        if i:
            mj.mju_subQuat(angular,wrist,old_wrist)
            angular=kin.body('link_6').xmat.reshape(3,3)@angular/dt
        old_wrist=wrist
        result=sim.step(state,action,jaw_orientation=tip.new_tensor(wrist)[None],
                        jaw_angular_velocity=tip.new_tensor(angular)[None])
        state=result.state
        summary.append([i*dt,float(result.table_clearance),float(result.support_N),float(state.velocity.norm()),
                        float(state.contacts[0,:,ContactState.CAPTURE].min()),float(state.contacts[0,:,0].min())])
    wall=time.perf_counter()-start
    np.savez_compressed(out,**frames,audit_2ms=summary,weights=sim.payload_mount_weights[0].numpy())
    print(f"Replayed {len(q) - 1} steps, final clearance {summary[-1][1]:.9f} m ({wall:.2f} s wall time)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.source, args.output)
