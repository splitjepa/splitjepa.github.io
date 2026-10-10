"""Task geometry and controlled ID/OOD environments."""

from dataclasses import dataclass
from typing import Any
import numpy as np
import sapien
import torch
import mani_skill.envs  # noqa: F401
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import REGISTERED_ENVS

def _np(x: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    object_attr: str
    goal_attr: str
    goal_is_planar: bool

    def object_position(self, env) -> np.ndarray:
        return _np(getattr(env.unwrapped, self.object_attr).pose.p).reshape(-1, 3)[0]

    def goal_position(self, env, obs: dict | None = None) -> np.ndarray:
        actor = getattr(env.unwrapped, self.goal_attr, None)
        if actor is not None:
            return _np(actor.pose.p).reshape(-1, 3)[0]
        if obs is not None and "goal_pos" in obs.get("extra", {}):
            return _np(obs["extra"]["goal_pos"]).reshape(-1, 3)[0]
        raise AttributeError(f"{self.task_id}: no {self.goal_attr} actor or extra.goal_pos")

    def distance(self, obj: np.ndarray, goal: np.ndarray) -> float:
        dims = 2 if self.goal_is_planar else 3
        return float(np.linalg.norm(np.asarray(obj)[:dims] - np.asarray(goal)[:dims]))


TASK_SPECS = {
    "PickCube-v1": TaskSpec(
        task_id="PickCube-v1",
        object_attr="cube",
        goal_attr="goal_site",
        goal_is_planar=False,
    ),
    "PushCube-v1": TaskSpec(
        task_id="PushCube-v1",
        object_attr="obj",
        goal_attr="goal_region",
        goal_is_planar=True,
    ),
}


def get_task_spec(task_id: str) -> TaskSpec:
    try:
        return TASK_SPECS[task_id]
    except KeyError as exc:
        raise KeyError(f"unsupported task {task_id!r}; supported={sorted(TASK_SPECS)}") from exc

# ManiSkill's installed PushCube-v1 defaults.  Keep these explicit so the OOD
# diagnostic changes exactly one nuisance around the official sensor/lighting
# configuration instead of silently changing target, FOV, and pose together.
PUSH_OFFICIAL_CAMERA = dict(
    eye=[0.3, 0.0, 0.6], target=[-0.1, 0.0, 0.1], fov=np.pi / 2,
    shader_pack="minimal",
)
PUSH_CAMERA_YAW_DEG = {
    "ood_camera_yaw_m10": -10.0,
    "ood_camera_yaw_p10": 10.0,
}
PUSH_LIGHTING_SCALE = {
    "ood_lighting_scale_0p6": 0.6,
}
PUSH_COMBINED = {
    "ood_visual_yaw_m10_light_0p6": (-10.0, 0.6),
    "ood_visual_yaw_p10_light_0p6": (10.0, 0.6),
}


def _camera_cfg(uid: str, eye, target, width=128, height=128, fov=np.pi / 2,
                shader=PUSH_OFFICIAL_CAMERA["shader_pack"]) -> CameraConfig:
    """Build a camera while preserving ManiSkill's official sensor renderer."""
    return CameraConfig(uid=uid, pose=sapien_utils.look_at(eye=np.asarray(eye, dtype=float),
                                                           target=np.asarray(target, dtype=float)),
                        width=width, height=height, fov=fov, near=0.01, far=100, shader_pack=shader)


def push_camera_at_yaw(yaw_deg: float) -> dict:
    """Rotate the official PushCube camera around its official look target."""
    eye = np.asarray(PUSH_OFFICIAL_CAMERA["eye"], dtype=float)
    target = np.asarray(PUSH_OFFICIAL_CAMERA["target"], dtype=float)
    rel = eye - target
    theta = np.deg2rad(float(yaw_deg))
    rotated = np.asarray([
        np.cos(theta) * rel[0] - np.sin(theta) * rel[1],
        np.sin(theta) * rel[0] + np.cos(theta) * rel[1],
        rel[2],
    ])
    return dict(eye=(target + rotated).tolist(), target=target.tolist(),
                fov=float(PUSH_OFFICIAL_CAMERA["fov"]),
                shader_pack=PUSH_OFFICIAL_CAMERA["shader_pack"])


def push_official_lighting(scale: float) -> dict:
    """Scale only the intensity of ManiSkill's official PushCube lighting."""
    scale = float(scale)
    return dict(
        ambient=(np.asarray([0.3, 0.3, 0.3]) * scale).tolist(),
        directionals=[
            dict(direction=[1.0, 1.0, -1.0], color=[scale] * 3, shadow=False),
            dict(direction=[0.0, 0.0, -1.0], color=[scale] * 3, shadow=False),
        ],
    )


def make_env(task: str, variant: str = "id", obs_mode: str = "rgb", control_mode: str = "pd_joint_pos",
             image_size: int = 128, sim_backend: str = "physx_cpu"):
    kwargs = dict(obs_mode=obs_mode, control_mode=control_mode, num_envs=1, sim_backend=sim_backend)
    if torch.cuda.is_available():
        # explicit CUDA renderer (the auto default would fall back to 'cpu',
        # which requires a Vulkan CPU ICD and usually fails on these nodes)
        kwargs["render_backend"] = "sapien_cuda"
    env_cls = REGISTERED_ENVS[task].cls

    if variant == "id":
        return env_cls(**kwargs)

    if variant in PUSH_CAMERA_YAW_DEG:
        camera = push_camera_at_yaw(PUSH_CAMERA_YAW_DEG[variant])

        class _PushOODCamEnv(env_cls):
            @property
            def _default_sensor_configs(self):
                return [_camera_cfg("base_camera", camera["eye"], camera["target"],
                                    width=image_size, height=image_size, fov=camera["fov"])]

        _PushOODCamEnv.__name__ = f"{env_cls.__name__}OfficialYawOOD"
        return _PushOODCamEnv(**kwargs)

    if variant in PUSH_LIGHTING_SCALE:
        return _with_lighting(env_cls, push_official_lighting(PUSH_LIGHTING_SCALE[variant]))(**kwargs)

    if variant in PUSH_COMBINED:
        yaw, scale = PUSH_COMBINED[variant]
        camera = push_camera_at_yaw(yaw)

        class _PushOODVisualEnv(_with_lighting(env_cls, push_official_lighting(scale))):
            @property
            def _default_sensor_configs(self):
                return [_camera_cfg("base_camera", camera["eye"], camera["target"],
                                    width=image_size, height=image_size, fov=camera["fov"])]

        _PushOODVisualEnv.__name__ = f"{env_cls.__name__}OfficialVisualOOD"
        return _PushOODVisualEnv(**kwargs)

    raise ValueError(f"unknown eval variant: {variant}")


def _with_lighting(env_cls, lighting: dict):
    class _OODLightingEnv(env_cls):
        def _load_lighting(self, options):
            amb = lighting.get("ambient", [0.05, 0.05, 0.08])
            self.scene.set_ambient_light(list(amb))
            for d in lighting["directionals"]:
                self.scene.add_directional_light(
                    d["direction"], d["color"], shadow=d.get("shadow", False),
                    shadow_scale=5, shadow_map_size=2048,
                )

    _OODLightingEnv.__name__ = f"{env_cls.__name__}OODLighting"
    return _OODLightingEnv


def obs_image(obs: dict, camera: str = "base_camera") -> np.ndarray:
    """Extract (1, H, W, 3) uint8 rgb from a ManiSkill obs dict."""
    return obs["sensor_data"][camera]["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
