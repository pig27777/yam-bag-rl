"""Three-vertex native contact geometry for the legacy mask regression."""

from xml.etree import ElementTree as ET

import numpy as np

from yam_bag_rl.robot.abc_workcell import CORNERS, PICK, add_layered_contact_support, add_workcell


class ContactFixture:
    payload_body = "payload_test"
    payload_geom = "core_test"
    half = np.array([.044, .034, .012])
    support_solref = np.array([.001, 1.])

    def add_scene(self, root):
        add_workcell(root)
        add_layered_contact_support(root, solref=".001 1")
        world = root.find("worldbody")
        film = ET.SubElement(world, "flexcomp", name="mask_triangle", type="direct", dim="2", mass=".003",
                             radius=".0005", point="-.07 -.25 .04 .01 -.25 .04 -.03 -.18 .04", element="0 1 2")
        ET.SubElement(film, "edge", damping=".001")
        ET.SubElement(film, "contact", contype="2", conaffinity="15")
        ET.SubElement(root.find("asset"), "mesh", name="core_box", vertex=" ".join(map(str, (CORNERS * self.half).ravel())))
        body = ET.SubElement(world, "body", name=self.payload_body, pos=" ".join(map(str, PICK + [0, 0, .015])))
        ET.SubElement(body, "freejoint", name="payload_free")
        ET.SubElement(body, "geom", name=self.payload_geom, type="mesh", mesh="core_box", mass=".2",
                      contype="8", conaffinity="2")
