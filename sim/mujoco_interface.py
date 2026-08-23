import sys
from typing import Optional, Tuple

import numpy as np

try:
    import mujoco
except ImportError:
    sys.exit("mujoco is not installed:  pip install mujoco")

from config.parameters import RobotConfig
from kinematics.leg_ik import quat_to_mat


class MujocoRobot:

    def __init__(self, cfg: RobotConfig):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(cfg.xml_path)
        self.data = mujoco.MjData(self.model)

        self.torso_id = self._body_id(cfg.torso_body)
        if self.torso_id < 0:
            sys.exit(f"ERROR: body '{cfg.torso_body}' not found.")

        if self.model.nu != len(cfg.joint_order):
            sys.exit(f"ERROR: model has {self.model.nu} actuators, expected "
                     f"{len(cfg.joint_order)}. Run tools/build_model.py.")

        self.qadr, self.ctrl_idx = [], []
        for name in cfg.joint_order:
            jid = self._id(mujoco.mjtObj.mjOBJ_JOINT, name)
            aid = self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if jid < 0 or aid < 0:
                sys.exit(f"ERROR: joint/actuator '{name}' not found.")
            self.qadr.append(int(self.model.jnt_qposadr[jid]))
            self.ctrl_idx.append(int(aid))
        self.qadr = np.array(self.qadr)
        self.ctrl_idx = np.array(self.ctrl_idx)

        self.ankle_roll_jid = self._id(mujoco.mjtObj.mjOBJ_JOINT, cfg.joint_order[5])

        self.ball_qadr = self.ball_dofadr = -1
        self.ball_mass = 0.0
        bjid = self._id(mujoco.mjtObj.mjOBJ_JOINT, cfg.ball_joint)
        if bjid >= 0:
            self.ball_qadr = int(self.model.jnt_qposadr[bjid])
            self.ball_dofadr = int(self.model.jnt_dofadr[bjid])
            self.ball_mass = float(self.model.body_mass[self._body_id(cfg.ball_body)])

        self.robot_mass = float(self.model.body_subtreemass[self.torso_id])
        if self.robot_mass < cfg.min_robot_mass:
            sys.exit("ERROR: robot mass is ~0. Placeholder inertials still present, "
                     "run tools/build_model.py.")

    def _id(self, objtype, name) -> int:
        return mujoco.mj_name2id(self.model, objtype, name)

    def _body_id(self, name) -> int:
        return self._id(mujoco.mjtObj.mjOBJ_BODY, name)

    @property
    def time(self) -> float:
        return float(self.data.time)

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def base_height(self) -> float:
        return float(self.data.qpos[2])

    @property
    def base_rotation(self) -> np.ndarray:
        return quat_to_mat(self.data.qpos[3:7])

    @property
    def base_angular_velocity(self) -> np.ndarray:
        return self.data.qvel[3:6].copy()

    @property
    def n_contacts(self) -> int:
        return int(getattr(self.data, "ncon", -1))

    def com_state(self) -> Tuple[np.ndarray, np.ndarray]:
        mujoco.mj_subtreeVel(self.model, self.data)
        return (self.data.subtree_com[self.torso_id].copy(),
                self.data.subtree_linvel[self.torso_id].copy())

    def is_finite(self) -> bool:
        return bool(np.all(np.isfinite(self.data.qpos))
                    and np.all(np.isfinite(self.data.qvel)))

    def lowest_foot_point(self) -> float:
        zmin = np.inf
        for gi in range(self.model.ngeom):
            bname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                      self.model.geom_bodyid[gi])
            if bname not in self.cfg.foot_bodies:
                continue
            if self.model.geom_contype[gi] == 0 and self.model.geom_conaffinity[gi] == 0:
                continue
            gp = self.data.geom_xpos[gi]
            gR = self.data.geom_xmat[gi].reshape(3, 3)
            if self.model.geom_type[gi] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = self.model.geom_dataid[gi]
                a = self.model.mesh_vertadr[mid]
                n = self.model.mesh_vertnum[mid]
                v = self.model.mesh_vert[a:a + n]
                zmin = min(zmin, float(np.min((v @ gR.T)[:, 2] + gp[2])))
            else:
                zmin = min(zmin, float(gp[2] - self.model.geom_rbound[gi]))
        return zmin

    def measure_ankle_height(self) -> float:
        self.data.qpos[:] = 0.0
        self.data.qpos[2] = 0.30
        self.data.qpos[3] = 1.0
        self.forward()
        zmin = self.lowest_foot_point()
        ank_z = float(self.data.xanchor[self.ankle_roll_jid][2])
        return ank_z - zmin

    def set_pose(self, torso_pos, q_a, q_b):
        self.data.qpos[:] = 0.0
        self.data.qpos[0:3] = torso_pos
        self.data.qpos[3] = 1.0
        self.data.qpos[self.qadr[0]:self.qadr[0] + 6] = q_a
        self.data.qpos[self.qadr[6]:self.qadr[6] + 6] = q_b
        self.forward()

    def settle_on_ground(self):
        self.data.qpos[2] -= self.lowest_foot_point()
        self.forward()

    def send(self, q_a, q_b):
        self.data.ctrl[self.ctrl_idx[:6]] = q_a
        self.data.ctrl[self.ctrl_idx[6:]] = q_b

    def forward(self):
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def clear_push(self):
        self.data.xfrc_applied[self.torso_id, :] = 0.0

    def apply_push(self, force: float, axis: str):
        if axis == "lat":
            self.data.xfrc_applied[self.torso_id, 0] = force
        else:
            self.data.xfrc_applied[self.torso_id, 1] = -force

    def push_direction(self, axis: str, force: float) -> np.ndarray:
        d = np.array([1.0, 0.0, 0.0]) if axis == "lat" else np.array([0.0, -1.0, 0.0])
        return d * (1.0 if force >= 0 else -1.0)

    def arrow_points(self, axis: str, force: float):
        p = self.data.xpos[self.torso_id].copy()
        d = self.push_direction(axis, force)
        length = 0.10 + 0.010 * min(abs(force), 40.0)
        return p - d * length, p + d * 0.05, d

    def launch_projectile(self, axis: str, force: float, duration: float,
                          distance: float = 0.60) -> Optional[float]:
        if self.ball_qadr < 0:
            return None
        d = self.push_direction(axis, force)
        v = abs(force) * duration / max(1e-6, self.ball_mass)
        self.data.qpos[self.ball_qadr:self.ball_qadr + 3] = \
            self.data.xpos[self.torso_id] - d * distance
        self.data.qpos[self.ball_qadr + 3:self.ball_qadr + 7] = [1, 0, 0, 0]
        self.data.qvel[self.ball_dofadr:self.ball_dofadr + 3] = d * v
        self.data.qvel[self.ball_dofadr + 3:self.ball_dofadr + 6] = 0.0
        return v


def add_arrow(scn, p_from, p_to, width, rgba):
    if scn is None or scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW,
                        np.zeros(3), np.zeros(3), np.zeros(9),
                        np.asarray(rgba, dtype=np.float32))
    a, b = np.asarray(p_from, float), np.asarray(p_to, float)
    try:
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, a, b)
    except AttributeError:
        mujoco.mjv_makeConnector(g, mujoco.mjtGeom.mjGEOM_ARROW, width,
                                 a[0], a[1], a[2], b[0], b[1], b[2])
    scn.ngeom += 1


def draw_push(scn, robot: MujocoRobot, axis: str, force: float,
              alpha: float, reset: bool = True):
    if scn is None:
        return
    if reset:
        scn.ngeom = 0
    if alpha <= 0.0:
        return
    a, b, _ = robot.arrow_points(axis, force)
    add_arrow(scn, a, b, 0.012, [1.0, 0.15, 0.1, float(alpha)])
    if scn.ngeom < scn.maxgeom:
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.018, 0.0, 0.0]), b, np.eye(3).flatten(),
                            np.array([1.0, 0.6, 0.1, float(alpha)], dtype=np.float32))
        scn.ngeom += 1


class VideoRecorder:

    def __init__(self, model, path: str, fps: int, width: int, height: int):
        self.path, self.fps = path, fps
        try:
            self.renderer = mujoco.Renderer(model, height=height, width=width)
        except Exception as e:
            sys.exit(f"Renderer failed: {e}\nResolution may exceed offwidth/offheight "
                     f"declared in the XML.")
        self.kind, self.writer = self._open(path, fps)

    @staticmethod
    def _open(path, fps):
        try:
            import imageio.v2 as imageio
            return "imageio", imageio.get_writer(path, fps=fps, macro_block_size=None,
                                                 codec="libx264", quality=8)
        except Exception as e1:
            try:
                import cv2
                return "cv2", [cv2, None]
            except Exception as e2:
                sys.exit("Video writing needs:  pip install imageio imageio-ffmpeg\n"
                         f"  imageio: {e1}\n  opencv : {e2}")

    @property
    def scene(self):
        return self.renderer.scene

    def capture(self, data):
        self.renderer.update_scene(data)

    def write(self):
        frame = self.renderer.render()
        if self.kind == "imageio":
            self.writer.append_data(frame)
        else:
            cv2, vw = self.writer
            if vw is None:
                h, w = frame.shape[:2]
                vw = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     self.fps, (w, h))
                self.writer = [cv2, vw]
            vw.write(frame[:, :, ::-1])

    def close(self):
        if self.kind == "imageio":
            self.writer.close()
        else:
            _, vw = self.writer
            if vw is not None:
                vw.release()
        self.renderer.close()
