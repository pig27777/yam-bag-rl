"""Render a cosmetic bag shell on an archived, single-world Yam_B trajectory."""

import argparse
from pathlib import Path
from xml.etree import ElementTree as ET

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image

from yam_bag_rl.paths import PROJECT_ROOT
from yam_bag_rl.robot.abc_workcell import add_workcell
from yam_bag_rl.robot.yam_model import build_yam_model


WIDTH, HEIGHT = 960, 540
SIZE_X, SIZE_Y = 25, 19
PAD_TIP = np.array([0., 0., -.1444])


def xml_values(values):
    return " ".join(str(value) for value in np.asarray(values).ravel())


def bag_frame(centers):
    side = .5 * (centers[1] + centers[3] - centers[0] - centers[2])
    length = .5 * (centers[2] + centers[3] - centers[0] - centers[1])
    x = side / np.linalg.norm(side)
    normal = np.cross(side, length)
    normal /= np.linalg.norm(normal)
    return centers.mean(axis=0), np.stack((x, np.cross(normal, x), normal), axis=1)


class BagSkin:
    def __init__(self, public_geometry, initial_centers):
        self.half = public_geometry[9:11].astype(float)
        _, rotation = bag_frame(initial_centers)
        side = .5 * (initial_centers[1] + initial_centers[3] - initial_centers[0] - initial_centers[2])
        length = .5 * (initial_centers[2] + initial_centers[3] - initial_centers[0] - initial_centers[1])
        self.anchor_span = np.array([side @ rotation[:, 0], length @ rotation[:, 1]])
        x, y = np.meshgrid(np.linspace(-self.half[0], self.half[0], SIZE_X),
                           np.linspace(-self.half[1], self.half[1], SIZE_Y))
        uv = np.column_stack((x.ravel(), y.ravel()))
        # Chamfer the sheet corners without reducing the nominal length or width.
        round_corner = .012
        offset = np.maximum(abs(uv) - (self.half - round_corner), 0.)
        radius = np.maximum(np.linalg.norm(offset, axis=1, keepdims=True), round_corner)
        self.uv = uv - np.sign(uv) * offset * (1. - round_corner / radius)
        self.n = len(uv)
        self.perimeter = np.array([
            *range(SIZE_X), *range(2 * SIZE_X - 1, self.n, SIZE_X),
            *range(self.n - 2, self.n - SIZE_X - 1, -1),
            *range(self.n - 2 * SIZE_X, 0, -SIZE_X),
        ])
        faces = []
        for j in range(SIZE_Y - 1):
            for i in range(SIZE_X - 1):
                a = j * SIZE_X + i
                b, c, d = a + 1, a + SIZE_X, a + SIZE_X + 1
                faces.extend(((a, b, d), (a, d, c),
                              (a + self.n, d + self.n, b + self.n),
                              (a + self.n, c + self.n, d + self.n)))
        for a, b in zip(self.perimeter, np.roll(self.perimeter, -1), strict=True):
            faces.extend(((a, a + self.n, b), (b, a + self.n, b + self.n)))
        self.faces = faces
        block_uv = .5 * self.anchor_span * np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]])
        delta = (self.uv[:, None] - block_uv) / [.035, .030]
        self.block_weights = np.exp(-.5 * np.sum(delta**2, axis=2))
        self.block_weights /= self.block_weights.sum(axis=1, keepdims=True)
        edge = np.prod(np.clip((self.half - abs(self.uv)) / [.022, .018], 0., 1.), axis=1)
        self.body_weight = edge * edge * (3. - 2. * edge)
        self.rest_crease = .0011 * np.sin(190 * self.uv[:, 0] + 74 * self.uv[:, 1]) * np.sin(
            35 * self.uv[:, 1] + 1.) * self.body_weight
        self.grab_uv = None
        self.pull = 0.

    def attach_xml(self, root):
        add_workcell(root)
        visual = root.find("visual")
        if visual is None:
            visual = ET.SubElement(root, "visual")
        global_settings = visual.find("global")
        if global_settings is None:
            global_settings = ET.SubElement(visual, "global")
        global_settings.attrib.update(offwidth=str(WIDTH), offheight=str(HEIGHT))
        assets = root.find("asset")
        ET.SubElement(assets, "texture", type="skybox", builtin="gradient", width="256", height="256",
                      rgb1=".20 .24 .28", rgb2=".065 .085 .110")
        ET.SubElement(assets, "material", name="preview_pe_film", rgba=".52 .79 .82 .57",
                      specular=".83", shininess=".68")
        world = root.find("worldbody")
        ET.SubElement(world, "light", name="preview_side_light", pos="-.38 -.34 .70", dir=".3 .3 -1",
                      diffuse=".50 .53 .57", directional="true", castshadow="false")
        deformable = ET.SubElement(root, "deformable")
        rest = np.vstack((np.c_[self.uv, np.full(self.n, .008)],
                          np.c_[self.uv, np.full(self.n, -.008)]))
        surface = ET.SubElement(deformable, "skin", name="preview_bag_skin", group="4",
                                material="preview_pe_film", rgba=".52 .79 .82 .57", vertex=xml_values(rest),
                                face=xml_values(self.faces))
        for i, pos in enumerate(rest):
            name = f"preview_skin_vertex_{i}"
            ET.SubElement(world, "body", name=name, mocap="true", pos=xml_values(pos))
            ET.SubElement(surface, "bone", body=name, bindpos=xml_values(pos),
                          bindquat="1 0 0 0", vertid=str(i), vertweight="1")

    def vertices(self, centers, half_thickness, position, held):
        center, rotation = bag_frame(centers)
        base = center + np.c_[self.uv, np.zeros(self.n)] @ rotation.T
        local_heights = (centers - center) @ rotation[:, 2]
        normal_offsets = self.block_weights @ local_heights
        thickness = .0008 + (self.block_weights @ half_thickness + .002) * self.body_weight
        half_separation = thickness[:, None] * rotation[:, 2]
        middle = base + (normal_offsets + self.rest_crease)[:, None] * rotation[:, 2]

        if held and self.grab_uv is None:
            self.grab_uv = np.clip((position - center) @ rotation[:, :2], -.96 * self.half, .96 * self.half)
        if self.grab_uv is not None:
            self.pull = min(1., self.pull + .34) if held else max(0., self.pull - .30)
            delta = self.uv - self.grab_uv
            root = np.argmin((delta**2).sum(axis=1))
            radial = np.linalg.norm(delta / [.042, .047], axis=1)
            follow = self.pull * np.exp(-radial**2)
            lift = position - middle[root]
            # Taper the pinch through both plies and gather small pleats around it.
            pull_normal = rotation[:, 2]
            folds = .0016 * follow * np.sin(7. * np.arctan2(delta[:, 1], delta[:, 0]) + 10. * radial)
            middle += follow[:, None] * lift + folds[:, None] * pull_normal
            half_separation *= (1. - .94 * follow[:, None])
            if not held and self.pull == 0.:
                self.grab_uv = None

        upper, lower = middle + half_separation, middle - half_separation
        for sheet in (upper, lower):
            in_bin = (abs(sheet[:, 0] + .03) < .20) & (abs(sheet[:, 1] - .03) < .140)
            sheet[:, 2] = np.maximum(sheet[:, 2], np.where(in_bin, .0046, .00065))
        return np.vstack((upper, lower)), center, rotation


def add_contents(scene, centers, half, rotation):
    for i, center in enumerate(centers):
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX,
                           np.array([.020, .016, max(.002, half[i])]), center, rotation.ravel(),
                           np.array([.76, .45 + .048 * i, .20, 1.]))
        scene.ngeom += 1


def add_seam(scene, points):
    for start, end in zip(points, np.roll(points, -1, axis=0), strict=True):
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                           np.eye(3).ravel(), np.array([.22, .58, .62, .85]))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .00062, start, end)
        scene.ngeom += 1


def add_reference_tethers(scene, tips, centers):
    for j, tip in enumerate(tips):
        anchor = .5 * (centers[j] + centers[j + 2])
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                           np.eye(3).ravel(), np.array([.25, .70, .78, .9]))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .0012, tip, anchor)
        scene.ngeom += 1


def render(record, output, *, still_only=False):
    with np.load(record, allow_pickle=False) as sample:
        trace, public = sample["trace"], sample["public_geometry"]
    bag = BagSkin(public, trace[0, 8:20].reshape(4, 3))
    _, model = build_yam_model(add_scene=bag.attach_xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    ids = [model.body(f"preview_skin_vertex_{i}").mocapid[0] for i in range(2 * bag.n)]
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [-.095, -.145, .118]
    camera.distance = .75
    camera.azimuth = 150.
    camera.elevation = -25.
    skin_opt = mujoco.MjvOption()
    skin_opt.skingroup[4] = 1
    skin_opt.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = True
    bare_opt = mujoco.MjvOption()
    bare_opt.flags[mujoco.mjtVisFlag.mjVIS_SKIN] = False
    pads = [model.body(f"{side}_pad").id for side in ("left", "right")]
    peak = int(trace[:, 28].argmax())
    encoder = None
    if not still_only:
        encoder = imageio_ffmpeg.write_frames(str(output / "yam_surrogate_skin_demo.mp4"), (WIDTH, HEIGHT),
                                             fps=25, codec="libx264", pix_fmt_in="rgb24", quality=8,
                                             output_params=["-movflags", "+faststart"], macro_block_size=2)
        encoder.send(None)
    try:
        for i, frame in enumerate(trace):
            data.qpos[:] = frame[:model.nq]
            mujoco.mj_kinematics(model, data)
            tips = [data.xpos[j] + data.xmat[j].reshape(3, 3) @ PAD_TIP for j in pads]
            centers = frame[8:20].reshape(4, 3)
            vertices, center, rotation = bag.vertices(centers, frame[20:24], np.mean(tips, axis=0), frame[29] > 0)
            if still_only and i != peak:
                continue
            data.mocap_pos[ids] = vertices
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera, scene_option=skin_opt)
            add_contents(renderer.scene, centers, frame[20:24], rotation)
            middle = .5 * (vertices[bag.perimeter] + vertices[bag.perimeter + bag.n])
            add_seam(renderer.scene, middle)
            pixels = renderer.render()
            if encoder is not None:
                encoder.send(pixels)
            if i == peak:
                close = mujoco.MjvCamera()
                close.lookat[:] = .66 * center + .34 * np.mean(tips, axis=0)
                close.distance, close.azimuth, close.elevation = .51, camera.azimuth, -16.
                renderer.update_scene(data, close, scene_option=skin_opt)
                add_contents(renderer.scene, centers, frame[20:24], rotation)
                add_seam(renderer.scene, middle)
                skinned = Image.fromarray(renderer.render().copy())
                renderer.update_scene(data, close, scene_option=bare_opt)
                add_contents(renderer.scene, centers, frame[20:24], rotation)
                if frame[29] > 0:
                    add_reference_tethers(renderer.scene, tips, centers)
                bare = Image.fromarray(renderer.render().copy())
                comparison = Image.new("RGB", (2 * WIDTH, HEIGHT))
                comparison.paste(bare, (0, 0))
                comparison.paste(skinned, (WIDTH, 0))
                comparison.save(output / "yam_surrogate_skin_comparison.png")
                if still_only:
                    break
    finally:
        if encoder is not None:
            encoder.close()
        renderer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    default = PROJECT_ROOT / "runs" / "surrogate-v1-large-scale"
    parser.add_argument("--record", type=Path, default=default / "c0_demo_trajectory.npz")
    parser.add_argument("--output", type=Path, default=default)
    parser.add_argument("--still-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    render(args.record, args.output, still_only=args.still_only)
