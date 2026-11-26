"""Rigid software fixture for the standalone macro-environment smoke test."""

from xml.etree import ElementTree as ET

import mujoco
import numpy as np

from ..robot.abc_workcell import PICK, add_workcell, box


class RigidScene:
    payload_body = "payload"
    payload_geom = "payload_geom"
    config = {}
    compile_model = staticmethod(mujoco.MjModel.from_xml_string)
    initialize = None
    is_flex = False
    half = np.array([.026, .024, .0325])

    @staticmethod
    def add_scene(root):
        add_workcell(root)
        obj = ET.SubElement(root.find("worldbody"), "body", name="payload", pos=" ".join(map(str, PICK + [0, 0, .0325])))
        ET.SubElement(obj, "freejoint", name="payload_free")
        box(obj, "payload_geom", [0, 0, 0], RigidScene.half, rgba=".80 .44 .19 1",
            mass=".12", friction="1 .005 .0005")

    def episode(self, model, initial, rng, scripted):
        state = initial.copy()
        addr = model.joint("payload_free").qposadr[0]
        xy = np.zeros(2) if scripted else rng.uniform([-.023, -.017], [.023, .017])
        yaw = 0. if scripted else rng.uniform(-.6, .6)
        state[addr:addr + 2] += xy
        state[addr + 3:addr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        return state, dict(payload_xy_offset_m=xy.tolist(), payload_yaw_rad=yaw, payload_mass_kg=.12)
