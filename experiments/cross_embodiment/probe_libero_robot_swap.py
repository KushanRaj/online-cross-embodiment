"""Probe whether LIBERO tasks can instantiate non-Panda robosuite arms.

This is a capability smoke test for the cross-embodiment EEF-adapter track. It
does not run a learned policy. It checks whether a standard LIBERO BDDL task can
be created with another single-arm robot under the OSC_POSE controller, then
takes one zero action and one small EEF action.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np


def register_mounted_aliases() -> None:
    """Register Mounted{Robot} aliases expected by LIBERO tabletop tasks."""
    from robosuite.models.robots.manipulators import IIWA, Jaco, Kinova3, Sawyer, UR5e
    from robosuite.robots import ROBOT_CLASS_MAPPING
    from robosuite.robots.single_arm import SingleArm

    for base_cls in [Sawyer, Jaco, Kinova3, UR5e, IIWA]:
        name = f"Mounted{base_cls.__name__}"
        if name not in ROBOT_CLASS_MAPPING:
            # RobotModelMeta registers this class name in robosuite's model registry.
            type(name, (base_cls,), {})
            ROBOT_CLASS_MAPPING[name] = SingleArm


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument(
        "--robots",
        nargs="+",
        default=["Panda", "Jaco", "Kinova3", "UR5e", "IIWA", "Sawyer"],
    )
    parser.add_argument("--out-dir", default="run-artifacts/libero-robot-swap-probe")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--controller", default="OSC_POSE")
    parser.add_argument("--delta", type=float, default=0.25)
    args = parser.parse_args()

    register_mounted_aliases()

    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.controllers import load_controller_config

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark = get_benchmark(args.suite)()
    task = benchmark.get_task(args.task_id)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    controller_config = load_controller_config(default_controller=args.controller)

    results: list[dict[str, Any]] = []

    for robot in args.robots:
        record: dict[str, Any] = {
            "robot": robot,
            "suite": args.suite,
            "task_id": args.task_id,
            "task_name": task.name,
            "language": task.language,
            "bddl_file": str(bddl_file),
            "controller": args.controller,
            "controller_type": controller_config.get("type"),
        }
        env = None
        try:
            env = OffScreenRenderEnv(
                bddl_file_name=str(bddl_file),
                robots=[robot],
                controller=args.controller,
                camera_names=["agentview"],
                camera_heights=args.resolution,
                camera_widths=args.resolution,
                control_freq=args.control_freq,
                horizon=10,
            )
            obs = env.reset()
            low, high = env.env.action_spec
            zero_action = np.zeros_like(low)
            delta_action = np.zeros_like(low)
            delta_action[0] = args.delta

            reset_eef = np.asarray(obs.get("robot0_eef_pos", np.full(3, np.nan)))
            if "agentview_image" in obs:
                imageio.imwrite(out_dir / f"{robot}_agentview_reset.png", obs["agentview_image"])

            obs_zero, reward_zero, done_zero, _ = env.step(zero_action)
            zero_eef = np.asarray(obs_zero.get("robot0_eef_pos", np.full(3, np.nan)))
            obs_delta, reward_delta, done_delta, _ = env.step(delta_action)
            delta_eef = np.asarray(obs_delta.get("robot0_eef_pos", np.full(3, np.nan)))
            if "agentview_image" in obs_delta:
                imageio.imwrite(out_dir / f"{robot}_agentview_after_delta.png", obs_delta["agentview_image"])

            record.update(
                {
                    "ok": True,
                    "action_dim": int(low.shape[0]),
                    "action_low": low,
                    "action_high": high,
                    "reset_eef": reset_eef,
                    "zero_eef": zero_eef,
                    "delta_eef": delta_eef,
                    "zero_eef_delta": zero_eef - reset_eef,
                    "commanded_delta_action": delta_action,
                    "observed_delta_eef_after_delta": delta_eef - zero_eef,
                    "reward_zero": reward_zero,
                    "done_zero": done_zero,
                    "reward_delta": reward_delta,
                    "done_delta": done_delta,
                    "obs_keys": sorted(obs.keys()),
                }
            )
        except Exception as exc:  # noqa: BLE001 - probe should report all failures.
            record.update({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})
        finally:
            if env is not None:
                env.close()
        results.append(to_jsonable(record))

    with (out_dir / "summary.json").open("w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
