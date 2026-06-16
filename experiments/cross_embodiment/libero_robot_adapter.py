"""LIBERO robot-swap helpers for EEF/OSC_POSE experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CAMERA_NAMES = ["agentview", "robot0_eye_in_hand"]


def register_mounted_aliases() -> None:
    """Register Mounted{Robot} aliases expected by LIBERO tabletop tasks."""
    from robosuite.models.robots.manipulators import IIWA, Jaco, Kinova3, Sawyer, UR5e
    from robosuite.robots import ROBOT_CLASS_MAPPING
    from robosuite.robots.single_arm import SingleArm

    for base_cls in [Sawyer, Jaco, Kinova3, UR5e, IIWA]:
        name = f"Mounted{base_cls.__name__}"
        if name not in ROBOT_CLASS_MAPPING:
            type(name, (base_cls,), {})
            ROBOT_CLASS_MAPPING[name] = SingleArm


def make_libero_env(
    task: Any,
    *,
    resolution: int,
    robot: str = "Panda",
    controller: str = "OSC_POSE",
    camera_names: list[str] | None = None,
    seed: int | None = None,
):
    """Create a LIBERO env with an explicit robot/controller contract."""
    register_mounted_aliases()

    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        robots=[robot],
        controller=controller,
        camera_names=camera_names or DEFAULT_CAMERA_NAMES,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    if seed is not None:
        env.seed(seed)
    return env, task_description


def _joint_addr(model, name: str, kind: str):
    addr = getattr(model, f"get_joint_{kind}_addr")(name)
    if isinstance(addr, tuple):
        return slice(int(addr[0]), int(addr[1]))
    return int(addr)


def _object_joint_names(model) -> list[str]:
    return [
        name
        for name in model.joint_names
        if not name.startswith("robot") and not name.startswith("gripper")
    ]


def copy_object_state(source_env, target_env) -> None:
    """Copy object/free-joint state by name from source env into target env."""
    source_model = source_env.sim.model
    target_model = target_env.sim.model
    shared_names = set(_object_joint_names(source_model)).intersection(_object_joint_names(target_model))

    for name in sorted(shared_names):
        src_qpos = _joint_addr(source_model, name, "qpos")
        dst_qpos = _joint_addr(target_model, name, "qpos")
        src_value = np.asarray(source_env.sim.data.qpos[src_qpos])
        dst_value = np.asarray(target_env.sim.data.qpos[dst_qpos])
        if src_value.shape == dst_value.shape:
            target_env.sim.data.qpos[dst_qpos] = src_value

        src_qvel = _joint_addr(source_model, name, "qvel")
        dst_qvel = _joint_addr(target_model, name, "qvel")
        src_vel = np.asarray(source_env.sim.data.qvel[src_qvel])
        dst_vel = np.asarray(target_env.sim.data.qvel[dst_qvel])
        if src_vel.shape == dst_vel.shape:
            target_env.sim.data.qvel[dst_qvel] = src_vel

    target_env.sim.forward()


def _attach_state_source_env(target_env, source_env) -> None:
    """Keep the Panda source env alive until the swapped target env closes.

    robosuite's offscreen EGL context can be invalidated if the temporary
    source env is closed while the target env is still rendering.
    """
    previous_source = getattr(target_env, "_libero_state_source_env", None)
    if previous_source is not None:
        previous_source.close()

    original_close = getattr(target_env, "_libero_original_close", target_env.close)
    target_env._libero_original_close = original_close
    target_env._libero_state_source_env = source_env

    def close_with_source():
        try:
            original_close()
        finally:
            attached = getattr(target_env, "_libero_state_source_env", None)
            if attached is not None:
                target_env._libero_state_source_env = None
                attached.close()

    target_env.close = close_with_source


def adapt_observation_for_libero_policy(obs: dict[str, Any]) -> dict[str, Any]:
    """Project swapped-robot observations to the Panda LIBERO policy contract.

    Cosmos/OpenPI LIBERO checkpoints were trained with Panda observations where
    ``robot0_gripper_qpos`` has width 2. Other robosuite grippers can expose
    more joint positions. For this adapter probe, keep EEF pose in the target
    robot's world frame but compress gripper proprio to a two-value opening
    proxy so model input width stays unchanged.
    """
    gripper_qpos = np.asarray(obs.get("robot0_gripper_qpos", []), dtype=np.float32).reshape(-1)
    if gripper_qpos.shape[0] <= 2:
        return obs

    adapted = dict(obs)
    opening = float(np.mean(gripper_qpos))
    adapted["robot0_gripper_qpos"] = np.asarray([opening, -opening], dtype=np.float32)
    if "robot0_gripper_qvel" in obs:
        gripper_qvel = np.asarray(obs["robot0_gripper_qvel"], dtype=np.float32).reshape(-1)
        velocity = float(np.mean(gripper_qvel)) if gripper_qvel.size else 0.0
        adapted["robot0_gripper_qvel"] = np.asarray([velocity, -velocity], dtype=np.float32)
    return adapted


def set_libero_initial_state_compatible(
    env,
    task: Any,
    initial_state: np.ndarray,
    *,
    resolution: int,
    robot: str = "Panda",
    controller: str = "OSC_POSE",
    camera_names: list[str] | None = None,
):
    """Set a LIBERO initial state even when the target robot is not Panda.

    LIBERO's stored init states are full simulator state vectors from the Panda
    benchmark env. Non-Panda robot swaps have different qpos/qvel widths, so a
    direct ``env.set_init_state`` fails. For robot swaps, this applies the Panda
    state to a temporary Panda env, then copies only named object joints into
    the target env and leaves the target robot in its native reset pose.
    """
    if robot == "Panda":
        return env.set_init_state(initial_state)

    source_env, _ = make_libero_env(
        task,
        resolution=resolution,
        robot="Panda",
        controller=controller,
        camera_names=camera_names,
    )
    source_env.reset()
    source_env.set_init_state(initial_state)
    copy_object_state(source_env, env)
    _attach_state_source_env(env, source_env)
    env.check_success()
    env._post_process()
    env._update_observables(force=True)
    return env.env._get_observations()
