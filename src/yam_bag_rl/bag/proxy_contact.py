"""Run the unchanged four-block payload and two-channel contact model at 2 ms."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import mujoco as mj

from .lumped_contact_proxy import ContactState, LumpedContactProxy, ProxyParams
from .macro_parcel_sim import MacroParcelSim, ParcelAction
from .neck_damping import NeckDampingParams
from .pad_torsion import PadTorsionParams
from .unified_payload import PayloadParamsX, UnifiedPayloadTemplate

from .replay_config import MONDAY, PAYLOAD

CASE = MONDAY["center_250g"]
CLOTH = np.asarray(CASE["cloth_size_m"])
CENTER = np.asarray(MONDAY["center_xy_m"])
DT = MONDAY["step_s"]


@torch.inference_mode()
def run(source, out, mass=CASE["mass_kg"], stiffness=CASE["rope_stiffness_N_m"], *,
        proxy_type=LumpedContactProxy):
    with np.load(source) as r:
        pads, wrist = r["pads"], r["wrist"]
    p = dict(MONDAY["proxy"])
    p.update(L0=[0.,0.],k=stiffness)
    as_tensor=lambda x:torch.as_tensor(x,dtype=torch.float32)
    params=PayloadParamsX.from_presets(PAYLOAD,["folded_clothes"],
       bag_half_extents=as_tensor(CLOTH[None]/2),total_mass=as_tensor([[mass]]),
       fill_ratio=as_tensor([[np.prod(CLOTH[:2])/(MONDAY["flat_bag_xy_m"][0]*MONDAY["flat_bag_xy_m"][1])]]))
    payload=UnifiedPayloadTemplate(params)
    sim=MacroParcelSim(proxy_type(ProxyParams.from_scalars(p),DT),payload=payload,
       neck_damping=NeckDampingParams(MONDAY["neck_damping"]),
       torsion=PadTorsionParams(radius_m=CASE["grip_radius_m"], neck_length_m=CASE["torsion_length_m"],
                               stiffness_Nm_rad=CASE["torsion_stiffness_Nm_rad"]))
    spacing = CASE["mount_half_spacing_m"]
    sim.payload_mount_weights = payload.weights_at(as_tensor([[[-spacing, 0], [spacing, 0]]]))
    state=sim.initial_state()
    shift=as_tensor([*CENTER,0.])
    state.payload.pos.add_(shift)
    state.com.add_(shift)
    samples={key:[] for key in ("time","centers","frame","half","indent","pads","vertical_force",
            "pinch_normal","capture","tension","support","clearance","com","velocity","slack")}
    history=[]
    surface_press=np.zeros(2)
    tension=np.zeros(2)
    support=mass*9.81
    clearance=0.
    t0=perf_counter()
    for i,pp in enumerate(pads):
        t=i*DT
        ppv=(pp-pads[max(0,i-1)])/DT
        line=pp[1]-pp[0]
        gap=np.linalg.norm(line)
        axis=line/gap
        inward=-((ppv-ppv.mean(0))*np.stack((-axis,axis))).sum(1)
        tip,tip_vel=as_tensor(pp[None]),as_tensor(ppv[None])
        st=state.payload
        R=payload.frame(st)
        local=torch.bmm(tip-st.pos.mean(1,keepdim=True),R)[0]
        inside=(local[:,:2].abs()<=params.bag_half_extents[0,:2]).all(1)
        previously=state.contacts[0,:,ContactState.CAPTURE]>.5
        nominal_depth=payload.half_height[0,0]-local[:,2]
        touch=inside & (nominal_depth>0) & (local[:,2]>-payload.half_height[0,0])
        # Calibrated finite gap / film compression. Capture is subject to
        # the existing gather, feed, normal and pull-out tests in the core.
        normal=(touch|previously) * (MONDAY["fold_normal_stiffness_N_m"]*max(0.,(MONDAY["fold_width_m"]-gap)/2))
        feed=nominal_depth.clamp_min(0)*touch
        normal_component=as_tensor(axis)@R[0,:,2]
        align=(1-normal_component.square()).clamp_min(0).sqrt()*touch
        if i%20==0:
            for k,v in dict(time=t,centers=st.pos[0],frame=R[0],half=payload.half_thickness(st.d_surf)[0],
                   indent=st.d_surf[0],pads=pp,vertical_force=surface_press,pinch_normal=normal,
                   capture=previously,tension=tension,support=support,clearance=clearance,
                   com=state.com[0],velocity=state.velocity[0],slack=state.contacts[0,:,ContactState.SLACK]).items():
                samples[k].append(np.asarray(v).copy())
        if i==len(pads)-1:break
        omega=np.zeros(3)
        if i:
            mj.mju_subQuat(omega,wrist[i],wrist[i-1])
            rr=np.empty(9)
            mj.mju_quat2Mat(rr,wrist[i])
            omega=rr.reshape(3,3)@omega/DT
        contact_weights=payload.weights_at(local[None,:,:2])
        weights_z=(torch.bmm(contact_weights,st.pos)*R[:,None,:,2]).sum(dim=-1)
        world_z=(tip*R[:,None,:,2]).sum(dim=-1)
        depth=payload.half_height-(world_z-weights_z)
        demand=payload.predict_surface_demand(st,tip+DT*tip_vel,inside[None])
        surface=payload.surface_contact(st,depth,contact_weights,inside[None],demand)
        surface_press=surface.pad_force_N[0].numpy().copy()
        action=ParcelAction(tip,tip_vel,normal[None],as_tensor(inward[None]),feed[None],align[None],
                            torch.zeros(1,3),torch.zeros(1,1))
        result=sim.step(state,action,jaw_orientation=as_tensor(wrist[i:i+1]),jaw_angular_velocity=as_tensor(omega[None]))
        state=result.state
        tension=result.tension[0].numpy().copy()
        support=float(result.support_N)
        clearance=float(result.table_clearance)
        history.append([t,clearance,support,float(state.velocity.norm()),*surface_press,*normal.numpy(),
           *state.contacts[0,:,ContactState.CAPTURE].numpy()])
        if i%250==0: print(t,"clearance",clearance,"pressure",surface_press,"tension",tension,flush=True)
    np.savez_compressed(out,**samples,history=np.asarray(history),steps=len(pads)-1,wall_s=perf_counter()-t0,
                        mass=mass,stiffness=stiffness)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("source",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("--mass",type=float,default=CASE["mass_kg"])
    p.add_argument("--stiffness",type=float,default=CASE["rope_stiffness_N_m"])
    a=p.parse_args()
    run(a.source,a.out,a.mass,a.stiffness)
