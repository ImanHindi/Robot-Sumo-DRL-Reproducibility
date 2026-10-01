from __future__ import annotations

import argparse
import re
import traceback

from controller import Robot

from common import get_agent_cfg, load_config
from runner import run_evaluation, run_frozen, run_scripted, run_static, run_training


def _infer_agent_id(robot: Robot, cli_agent_id: int | None) -> int:
    """Resolve the robot index even when a Webots PROTO drops controllerArgs.

    Priority:
      1) explicit --agent-id supplied by Webots/controllerArgs
      2) customData such as SUMO_0 / SUMO_1
      3) robot name such as sumo_red/sumo_blue or names ending in 0/1
    """
    if cli_agent_id is not None:
        return int(cli_agent_id)

    custom = str(robot.getCustomData() or "")
    match = re.search(r"(?:SUMO[_-]?)?(\d+)", custom, re.IGNORECASE)
    if match:
        return int(match.group(1))

    name = str(robot.getName() or "")
    lname = name.lower()
    if "red" in lname:
        return 0
    if "blue" in lname:
        return 1
    match = re.search(r"(\d+)$", name)
    if match:
        return int(match.group(1))

    raise RuntimeError(
        f"Cannot determine agent id for robot name={name!r}, customData={custom!r}. "
        "Set customData to SUMO_0/SUMO_1 or pass --agent-id."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-id", type=int, default=None)
    args, _ = parser.parse_known_args()

    # Construct Robot first so customData/name can be used as a fallback when
    # controllerArgs are unavailable in an older/custom PROTO.
    robot = Robot()
    agent_id = _infer_agent_id(robot, args.agent_id)

    cfg = load_config()
    agent_cfg = get_agent_cfg(cfg, agent_id)
    mode = str(agent_cfg.get("mode", "static")).lower()
    print(
        f"[sumo_research] agent={agent_id} mode={mode} "
        f"name={robot.getName()} customData={robot.getCustomData()}"
    )

    try:
        if mode == "train":
            run_training(robot, cfg, agent_id)
        elif mode in {"eval", "evaluate", "evaluation"}:
            run_evaluation(robot, cfg, agent_id)
        elif mode == "static":
            run_static(robot, cfg, agent_id)
        elif mode == "scripted":
            run_scripted(robot, cfg, agent_id)
        elif mode in {"frozen", "model"}:
            run_frozen(robot, cfg, agent_id)
        else:
            raise ValueError(f"Unknown agent mode: {mode}")
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
