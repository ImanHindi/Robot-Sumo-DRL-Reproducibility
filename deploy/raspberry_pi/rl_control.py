#!/usr/bin/env python3
"""Raspberry Pi inference for policies trained by the enhanced Webots package.

Important: the simulator uses normalized proximity values. Physical range measurements
are converted to an approximate Webots-like proximity signal before model inference.
The v1.12 professional robot has four downward edge sensors and four opponent range directions; the low-level MCU should
OR front+rear sensors on each side and send the same two left/right edge flags expected
by this six-dimensional policy interface. Perform hardware calibration before reporting
sim-to-real results.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import serial

DIRECTIONS = ["forward", "backward", "left", "right", "spin_left", "spin_right"]
SPEEDS = ["low", "medium", "high"]
ACTION_TABLE = [(d, s) for d in DIRECTIONS for s in SPEEDS]
SPEED_VALUE = {"low": 50.0, "medium": 80.0, "high": 100.0}


def load_model(algo, path):
    if algo == "A2C":
        from stable_baselines3 import A2C; return A2C.load(path)
    if algo == "PPO":
        from stable_baselines3 import PPO; return PPO.load(path)
    if algo == "DQN":
        from stable_baselines3 import DQN; return DQN.load(path)
    if algo == "RecurrentPPO":
        from sb3_contrib import RecurrentPPO; return RecurrentPPO.load(path)
    raise ValueError(algo)


def parse_sensor_line(line):
    parts = line.strip().split(",")
    if len(parts) != 7 or parts[0] != "S":
        raise ValueError(f"Bad sensor packet: {line!r}")
    return tuple(float(x) for x in parts[1:])


def range_cm_to_proximity(cm: float, max_training_range_cm: float) -> float:
    """Approximate the v1.12 Webots linear range-sensor lookup table."""
    return float(np.clip(1.0 - cm / max_training_range_cm, 0.0, 1.0))


def digital_obstacle_to_proximity(flag: float) -> float:
    return float(np.clip(flag, 0.0, 1.0))


def build_obs(front_cm, back_cm, left_cm, right_cm, line_l_flag, line_r_flag):
    # v1.12 reference ranges: front 85 cm, sides 70 cm, rear 65 cm.
    rear_prox = range_cm_to_proximity(back_cm, 65.0)
    return np.asarray([
        range_cm_to_proximity(front_cm, 85.0),
        float(rear_prox >= 0.10),
        range_cm_to_proximity(left_cm, 70.0),
        range_cm_to_proximity(right_cm, 70.0),
        digital_obstacle_to_proximity(line_l_flag),
        digital_obstacle_to_proximity(line_r_flag),
    ], dtype=np.float32)


def action_to_vel(action_idx):
    direction, speed_label = ACTION_TABLE[int(action_idx)]
    v = SPEED_VALUE[speed_label]
    if direction == "forward": left, right = v, v
    elif direction == "backward": left, right = -v, -v
    elif direction == "left": left, right = 0.5*v, v
    elif direction == "right": left, right = v, 0.5*v
    elif direction == "spin_left": left, right = -v, v
    else: left, right = v, -v
    return int(np.clip(left, -100, 100)), int(np.clip(right, -100, 100))


def edge_safety_override(obs, left, right, enabled=True):
    if not enabled:
        return left, right
    # Hardware flags should be calibrated so 1 indicates edge danger.
    if obs[4] > 0.5 or obs[5] > 0.5:
        return -70, -70
    return left, right


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--model", required=True)
    p.add_argument("--algorithm", required=True, choices=["A2C", "PPO", "DQN", "RecurrentPPO"])
    p.add_argument("--hz", type=float, default=25.0)
    p.add_argument("--disable-safety-override", action="store_true")
    args = p.parse_args()

    model = load_model(args.algorithm, args.model)
    ser = serial.Serial(args.port, args.baud, timeout=0.15)
    time.sleep(2.0)
    dt = 1.0 / args.hz
    lstm_state = None
    episode_start = np.ones((1,), dtype=bool)

    try:
        while True:
            ser.write(b"R\n")
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                ser.write(b"V,0,0\n"); time.sleep(dt); continue
            try:
                obs = build_obs(*parse_sensor_line(line))
            except Exception:
                continue

            if args.algorithm == "RecurrentPPO":
                action, lstm_state = model.predict(obs, state=lstm_state, episode_start=episode_start, deterministic=True)
                episode_start[:] = False
            else:
                action, _ = model.predict(obs, deterministic=True)
            left, right = action_to_vel(np.asarray(action).reshape(-1)[0])
            left, right = edge_safety_override(obs, left, right, not args.disable_safety_override)
            ser.write(f"V,{left},{right}\n".encode())
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        try: ser.write(b"V,0,0\n")
        except Exception: pass
        ser.close()


if __name__ == "__main__":
    main()
