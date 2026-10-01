from __future__ import annotations

import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CTRL = ROOT / "controllers" / "sumo_research"
sys.path.insert(0, str(CTRL))

from env import ACTION_TABLE, motion_flags, motor_command_for_action  # noqa: E402


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config" / "base_experiment.yaml").read_text(encoding="utf-8"))
    proto = (ROOT / "protos" / "SumoRobot2.proto").read_text(encoding="utf-8")

    assert len(ACTION_TABLE) == 18
    expected_dirs = ["forward"] * 3 + ["backward"] * 3 + ["left"] * 3 + ["right"] * 3 + ["spin_left"] * 3 + ["spin_right"] * 3
    assert [d for d, _ in ACTION_TABLE] == expected_dirs
    assert "Local forward direction is -Y" in proto
    assert proto.count("axis 1 0 0") >= 2

    print("[ACTION AUDIT v1.12] authoritative mapping")
    print(" id  action                  left(rad/s) right(rad/s)")
    print(" --  ----------------------  ----------- ------------")
    for idx, (direction, speed) in enumerate(ACTION_TABLE):
        _, _, left, right = motor_command_for_action(cfg, idx)
        print(f" {idx:2d}  {direction + '_' + speed:22s} {left:11.2f} {right:12.2f}")

    # Directional invariants. These catch the exact class of hidden mapping/sign
    # errors that motivated v1.11 before a long RL run is started.
    for idx in (0, 1, 2):
        _, _, l, r = motor_command_for_action(cfg, idx)
        assert l > 0 and r > 0 and abs(l - r) < 1e-12
    for idx in (3, 4, 5):
        _, _, l, r = motor_command_for_action(cfg, idx)
        assert l < 0 and r < 0 and abs(l - r) < 1e-12
    # Curved steering actions are still physically forward-translational because
    # both wheels rotate forward.  v1.12 reward/metrics must classify them that way.
    for idx in (6, 7, 8, 9, 10, 11):
        direction, _, l, r = motor_command_for_action(cfg, idx)
        flags = motion_flags(direction, l, r)
        assert flags["translational_forward"] and flags["curved_forward"]
        assert not flags["straight_forward"]
    for idx in (12, 13, 14):
        _, _, l, r = motor_command_for_action(cfg, idx)
        assert l < 0 < r
    for idx in (15, 16, 17):
        _, _, l, r = motor_command_for_action(cfg, idx)
        assert r < 0 < l

    print("[OK] action table, wheel signs, curved-forward classification, speed labels, and PROTO hinge convention are consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
