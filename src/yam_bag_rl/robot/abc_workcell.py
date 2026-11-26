"""Compose the project ABC workcell around the shared single Yam_B model."""

from copy import deepcopy
import json
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

from ..paths import CONFIGS
from .yam_model import ARM_JOINTS, RAILS


def _vec(x):
    return " ".join(map(str, x))


def _multiply(a, b):
    result = np.empty(4)
    mujoco.mju_mulQuat(result, np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return result


def _park_arm(root, base, orientation, pose, opening):
    # Bake the existing rigid model's scene transforms at the workcell's
    # parked posture. Yam_A retains its visual and physical collision shapes.
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    for name, value in zip((*ARM_JOINTS, *RAILS), (*pose, opening / 2, opening / 2), strict=True):
        data.joint(name).qpos[:] = value
    mujoco.mj_kinematics(model, data)
    movable = {b for b, n in enumerate(model.body_jntnum) if n}
    world = root.find("worldbody")
    parked = deepcopy(world.find("body[@name='arm']"))
    parked.set("pos", _vec(base))
    parked.set("quat", _vec(orientation))
    for body in parked.iter("body"):
        bid = model.body(body.attrib["name"]).id
        if bid in movable:
            parent = model.body_parentid[bid]
            parent_rotation = data.xmat[parent].reshape(3, 3)
            relative = parent_rotation.T @ (data.xpos[bid] - data.xpos[parent])
            inverse = np.r_[data.xquat[parent, 0], -data.xquat[parent, 1:]]
            body.set("pos", _vec(relative))
            body.set("quat", _vec(_multiply(inverse, data.xquat[bid])))
        for joint in body.findall("joint"):
            body.remove(joint)
    for node in parked.iter():
        if "name" in node.attrib:
            node.set("name", f"Yam_A_{node.attrib['name']}")
    parked.set("name", "Yam_A")
    world.append(parked)


def add_abc_workcell(root, scene, *, parked_arm=None):
    """Add ABC table, calibrated camera frames and optionally a fixed Yam_A."""
    world = root.find("worldbody")
    table = scene["table"]
    ET.SubElement(world, "geom", name="table", type="box", pos=_vec(table["center_m"]),
                  size=_vec(np.asarray(table["size_m"]) / 2), rgba=".38 .43 .47 1")
    top = scene["top_camera"]
    # Project camera frames are right/down/forward. MJCF cameras are
    # right/up/back, a local 180-degree rotation about x.
    optical_to_mj = [0, 1, 0, 0]
    ET.SubElement(world, "camera", name="abc_top", pos=_vec(top["position_m"]),
                  quat=_vec(_multiply(top["quat_wxyz"], optical_to_mj)),
                  fovy=str(top["vertical_fov_deg"]))

    wrist = scene["wrist_camera"]
    # Shared CAD puts the ABC gripper origin at link_6 and sends
    # (x, y, z) in the URDF gripper frame to (-y, -x, -z) there.
    rotation = np.sqrt(.5) * np.array([0., 1., -1., 0.])
    x, y, z = wrist["position_m"]
    quat = _multiply(_multiply(rotation, wrist["quat_wxyz"]), optical_to_mj)
    fovy = np.rad2deg(2 * np.arctan(wrist["sensor_size_m"][1] / (2 * wrist["focal_length_m"])))
    link = world.find(".//body[@name='link_6']")
    ET.SubElement(link, "camera", name="abc_wrist_b", pos=_vec((-y, -x, -z)), quat=_vec(quat), fovy=str(fovy))
    if parked_arm is not None:
        index, joint_angles = parked_arm
        _park_arm(root, scene["arm_bases_m"][index], scene["arm_base_quat_wxyz"],
                  joint_angles, -2 * scene["ready_rail_m"])


ABC = json.loads((CONFIGS / "workstation/abc.json").read_text())
BIN = ABC["bin"]
PICK = np.asarray(ABC["pick_center_m"])
CORNERS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])


def box(parent, name, pos, half, **attrs):
    return ET.SubElement(parent, "geom", name=name, pos=_vec(pos), size=_vec(half), type="box", **attrs)


def add_workcell(root):
    park = ABC["parked_arm"]
    add_abc_workcell(root, ABC, parked_arm=(park["abc_index"], park["ready_joints_rad"]))
    world = root.find("worldbody")
    center = np.array(BIN["center_m"])
    hx, hy = BIN["inner_half_xy_m"]
    t = BIN["wall_thickness_m"]
    z = BIN["rim_height_m"] / 2
    box(world, "bin_floor", center + [0, 0, .002], [hx, hy, .002], rgba=".16 .35 .46 1")
    for k, (x, y) in enumerate(((hx, 0), (-hx, 0), (0, hy), (0, -hy))):
        half = [t / 2, hy + t, z] if x else [hx, t / 2, z]
        box(world, f"bin_wall_{k}", center + [x, y, z], half, rgba=".20 .44 .56 1")


def add_layered_contact_support(root, *, solref):
    """Separate native soft-surface support from rigid workcell contact."""
    world = root.find("worldbody")
    world.find("geom[@name='table']").attrib.update(contype="16", conaffinity="9")
    ET.SubElement(world, "geom", name="table_flex_support", type="plane", size="1 1 .05",
                  contype="0", conaffinity="2", rgba="0 0 0 0", solref=solref)
    root.find(".//default[@class='collision']/geom").set("conaffinity", "1")
