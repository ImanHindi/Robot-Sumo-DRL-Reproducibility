from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium import Env, spaces

from common import drain_json, resolve_run_dir, send_json

DIRECTIONS = ["forward", "backward", "left", "right", "spin_left", "spin_right"]
SPEEDS = ["low", "medium", "high"]
ACTION_TABLE = [(d, s) for d in DIRECTIONS for s in SPEEDS]
DEFAULT_SPEED_VALUE = {"low": 7.0, "medium": 12.0, "high": 18.0}


def action_speed(cfg: Dict[str, Any], speed_label: str) -> float:
    speeds = cfg.get("drive", {}).get("action_speeds_rad_s", DEFAULT_SPEED_VALUE)
    return float(speeds.get(speed_label, DEFAULT_SPEED_VALUE[speed_label]))


def max_drive_speed(cfg: Dict[str, Any]) -> float:
    return float(cfg.get("drive", {}).get("motor_max_velocity_rad_s", 22.0))


def motor_command_for_action(cfg: Dict[str, Any], action: int) -> tuple[str, str, float, float]:
    """Authoritative v1.12 discrete action -> wheel-command mapping (unchanged from v1.11).

    Robot local forward is -Y and the PROTO is unchanged from v1.10. Positive
    velocity on both wheels therefore drives the wedge/front in the -Y direction.
    Keeping this mapping in one helper lets the runtime, frozen-policy controller,
    and static audit script share exactly the same convention.
    """
    idx = int(np.asarray(action).reshape(-1)[0])
    if idx < 0 or idx >= len(ACTION_TABLE):
        raise IndexError(f"action {idx} outside [0,{len(ACTION_TABLE)-1}]")
    direction, speed_label = ACTION_TABLE[idx]
    v = action_speed(cfg, speed_label)
    if direction == "forward":
        left, right = v, v
    elif direction == "backward":
        left, right = -v, -v
    elif direction == "left":
        left, right = 0.5 * v, v
    elif direction == "right":
        left, right = v, 0.5 * v
    elif direction == "spin_left":
        left, right = -v, v
    elif direction == "spin_right":
        left, right = v, -v
    else:  # defensive only; ACTION_TABLE is fixed above
        left = right = 0.0
    return direction, speed_label, float(left), float(right)


def action_name(action: int) -> str:
    d, s = ACTION_TABLE[int(action)]
    return f"{d}_{s}"


def motion_flags(direction: str, left: float, right: float, eps: float = 1e-9) -> Dict[str, bool]:
    """Classify wheel commands by physical translation, not action label alone.

    In this differential-drive action set, ``left`` and ``right`` steering actions
    still command both wheels forward.  v1.12 therefore distinguishes straight
    forward from curved forward while treating both as forward-translational
    motion for attack/contact reward and behavioral metrics.
    """
    lf = float(left) > eps
    rf = float(right) > eps
    lb = float(left) < -eps
    rb = float(right) < -eps
    translational_forward = lf and rf
    translational_backward = lb and rb
    straight_forward = translational_forward and direction == "forward"
    curved_forward = translational_forward and direction in {"left", "right"}
    spin = (lf and rb) or (lb and rf)
    return {
        "translational_forward": translational_forward,
        "straight_forward": straight_forward,
        "curved_forward": curved_forward,
        "translational_backward": translational_backward,
        "spin": spin,
    }


class WebotsSumoEnv(Env):
    """Single-agent Webots environment used by training and evaluation.

    Policy observation remains six-dimensional for deployment simplicity:
      [front proximity, rear obstacle flag, left proximity, right proximity,
       left-side edge flag, right-side edge flag]

    v1.12 keeps the v1.11 learning layer and corrects reset integrity plus translational-motion accounting. The world/PROTO/physics stay byte-identical
    to v1.10. Reward shaping uses *privileged training-only telemetry* from the
    Supervisor (center distance and real robot-robot contact). The policy never sees
    that telemetry in its observation, so inference still depends only on hardware-
    reproducible sensors.
    """

    metadata = {"render_modes": []}

    def __init__(self, robot, cfg: Dict[str, Any], agent_id: int, phase: str = "train"):
        super().__init__()
        self.robot = robot
        self.cfg = cfg
        self.agent_id = int(agent_id)
        self.phase = phase
        self.timestep = int(robot.getBasicTimeStep())

        arena = cfg["arena"]
        env_cfg = cfg["environment"]
        self.max_episode_steps = int(arena["max_episode_steps"])
        self.sensor_max = float(env_cfg["sensor_max"])
        # Kept for scripted-opponent compatibility; RL reward uses the stricter
        # front_attack_threshold below so "engagement" means genuinely attackable.
        self.front_engage_threshold = float(env_cfg["front_engage_threshold"])
        self.front_attack_threshold = float(env_cfg.get("front_attack_threshold", 0.38))
        self.side_detect_threshold = float(env_cfg["side_detect_threshold"])
        self.back_detect_threshold = float(env_cfg.get("back_detect_threshold", 0.15))
        self.target_detect_floor = float(env_cfg.get("target_detect_floor", 0.04))
        self.line_risk_low = float(env_cfg["line_risk_low"])
        self.line_risk_high = float(env_cfg["line_risk_high"])

        self.left_motor = robot.getDevice("left wheel motor")
        self.right_motor = robot.getDevice("right wheel motor")
        self.front_ds = robot.getDevice("front_distance_sensor")
        self.back_ds = robot.getDevice("back_sensor")
        self.side_left_ds = robot.getDevice("left_side_sensor")
        self.side_right_ds = robot.getDevice("right_side_sensor")
        self.line_left_front = robot.getDevice("left_ground_sensor")
        self.line_right_front = robot.getDevice("right_ground_sensor")
        self.line_left_rear = robot.getDevice("left_rear_ground_sensor")
        self.line_right_rear = robot.getDevice("right_rear_ground_sensor")
        self.emitter = robot.getDevice("emitter")
        self.receiver = robot.getDevice("receiver")
        self.receiver.enable(self.timestep)

        sensors = [
            self.front_ds, self.back_ds, self.side_left_ds, self.side_right_ds,
            self.line_left_front, self.line_right_front,
            self.line_left_rear, self.line_right_rear,
        ]
        for ds in sensors:
            ds.enable(self.timestep)

        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        self.action_space = spaces.Discrete(len(ACTION_TABLE))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)

        self.episode_id = 0
        self.episode_index = 0
        self.step_count = 0
        self.total_env_steps = 0
        self.episode_reward = 0.0
        self.curriculum_level = "unknown"
        self.curriculum_progress = 0.0
        self.start_wall = None
        self.last_outcome: Optional[Dict[str, Any]] = None

        self._reset_episode_counters()

        self.run_dir = resolve_run_dir(cfg, self.agent_id)
        self.csv_path = self.run_dir / f"{phase}_episodes_agent{self.agent_id}.csv"
        self._csv_header_written = self.csv_path.exists() and self.csv_path.stat().st_size > 0

        print(
            "[action_audit] v1.12 action_space=18 "
            "forward=[0,1,2] backward=[3,4,5] left=[6,7,8] right=[9,10,11] "
            "spin_left=[12,13,14] spin_right=[15,16,17]"
        )
        f0 = motor_command_for_action(cfg, 0)
        f2 = motor_command_for_action(cfg, 2)
        print(
            f"[action_audit] forward_low motors=({f0[2]:+.1f},{f0[3]:+.1f}) "
            f"forward_high motors=({f2[2]:+.1f},{f2[3]:+.1f}) local_forward=-Y"
        )

    def _reset_episode_counters(self) -> None:
        # v1.12 motion accounting: forward_ratio is the physically meaningful
        # translational-forward ratio (both wheels positive).  Straight and curved
        # forward motion are also logged separately for publication transparency.
        self.forward_steps = 0
        self.straight_forward_steps = 0
        self.curved_forward_steps = 0
        self.high_speed_steps = 0
        self.attack_steps = 0
        self.engagement_steps = 0
        self.edge_risk_steps = 0
        self.turn_toward_steps = 0
        self.search_forward_steps = 0
        self.scan_steps = 0
        self.target_detected_steps = 0
        self.approach_progress_steps = 0
        self.retreat_from_target_steps = 0
        self.backward_without_edge_steps = 0
        self.front_visible_nonforward_steps = 0
        self.front_visible_backward_steps = 0
        self.contact_steps_local = 0
        self.contact_forward_steps = 0
        self.first_contact_events = 0
        self.push_outward_progress_steps = 0
        self.action_counts = np.zeros(len(ACTION_TABLE), dtype=np.int64)

        self.prev_target_signal = 0.0
        self.latest_distance_m: Optional[float] = None
        self.prev_distance_m: Optional[float] = None
        self.latest_contact = False
        self.prev_contact_state = False
        self.contact_started_this_step = False
        self.latest_own_radius_m: Optional[float] = None
        self.latest_opponent_radius_m: Optional[float] = None
        self.prev_opponent_radius_m: Optional[float] = None

    def _normalize(self, value: float) -> float:
        return float(np.clip(value / self.sensor_max, 0.0, 1.0))

    def _line_risk(self, v: float) -> bool:
        return v < self.line_risk_low or v > self.line_risk_high

    def read_observation(self) -> np.ndarray:
        front = self._normalize(self.front_ds.getValue())
        back_raw = self._normalize(self.back_ds.getValue())
        left = self._normalize(self.side_left_ds.getValue())
        right = self._normalize(self.side_right_ds.getValue())

        lf = self._normalize(self.line_left_front.getValue())
        lr = self._normalize(self.line_left_rear.getValue())
        rf = self._normalize(self.line_right_front.getValue())
        rr = self._normalize(self.line_right_rear.getValue())

        back_flag = float(back_raw >= self.back_detect_threshold)
        left_edge_flag = float(self._line_risk(lf) or self._line_risk(lr))
        right_edge_flag = float(self._line_risk(rf) or self._line_risk(rr))
        return np.asarray(
            [front, back_flag, left, right, left_edge_flag, right_edge_flag],
            dtype=np.float32,
        )

    def _apply_action(self, action: int) -> tuple[int, str, str, float, float]:
        idx = int(np.asarray(action).reshape(-1)[0])
        direction, speed_label, left, right = motor_command_for_action(self.cfg, idx)
        self.left_motor.setVelocity(left)
        self.right_motor.setVelocity(right)
        self.action_counts[idx] += 1
        return idx, direction, speed_label, left, right

    def _process_messages(self) -> Optional[Dict[str, Any]]:
        outcome = None
        self.contact_started_this_step = False
        telemetry_seen = False
        for msg in drain_json(self.receiver):
            mtype = msg.get("type")
            if mtype == "reset_done":
                self.episode_id = int(msg.get("episode_id", self.episode_id))
            elif mtype == "training_telemetry":
                msg_ep = int(msg.get("episode_id", -1))
                if msg_ep != self.episode_id:
                    continue
                telemetry_seen = True
                contact = bool(msg.get("robot_contact", False))
                self.contact_started_this_step = contact and not self.prev_contact_state
                self.latest_contact = contact
                self.prev_contact_state = contact
                d = msg.get("distance_m")
                self.latest_distance_m = None if d is None else float(d)
                radii = msg.get("radii_m", {}) or {}
                own_key = str(self.agent_id)
                opp_key = str(1 - self.agent_id)
                own_r = radii.get(own_key)
                opp_r = radii.get(opp_key)
                self.latest_own_radius_m = None if own_r is None else float(own_r)
                self.latest_opponent_radius_m = None if opp_r is None else float(opp_r)
            elif mtype == "episode_end":
                msg_ep = int(msg.get("episode_id", -1))
                if msg_ep == self.episode_id:
                    send_json(self.emitter, {
                        "type": "episode_end_ack",
                        "agent_id": self.agent_id,
                        "episode_id": msg_ep,
                    })
                    outcome = msg
        if not telemetry_seen:
            # Do not invent a new contact transition if the packet is absent.
            self.contact_started_this_step = False
        return outcome

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        target_episode = self.episode_id + 1
        protocol = self.cfg.get("protocol", {})
        retry_every = max(1, int(protocol.get("reset_retry_interval_steps", 20)))
        max_wait_steps = max(retry_every + 1, int(protocol.get("reset_max_wait_steps", 2000)))
        waited_steps = 0
        reset_done_seen = False

        while self.robot.step(self.timestep) != -1:
            # Hold the learner physically stationary throughout prepare/settle/ack.
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)
            if waited_steps % retry_every == 0 and not reset_done_seen:
                send_json(self.emitter, {
                    "type": "reset_request",
                    "agent_id": self.agent_id,
                    "target_episode": target_episode,
                    "training_steps": int(self.total_env_steps),
                    "phase": self.phase,
                })

            for msg in drain_json(self.receiver):
                mtype = msg.get("type")
                if mtype == "reset_done" and int(msg.get("episode_id", -1)) == target_episode:
                    self.episode_id = target_episode
                    self.curriculum_level = str(msg.get("curriculum_level", "unknown"))
                    self.curriculum_progress = float(msg.get("curriculum_progress", 0.0) or 0.0)
                    reset_done_seen = True
                    # Acknowledge the verified pose, but keep motors at zero until
                    # the supervisor releases all controllers simultaneously.
                    send_json(self.emitter, {
                        "type": "reset_ack",
                        "agent_id": self.agent_id,
                        "episode_id": target_episode,
                    })
                    continue
                if mtype == "reset_release" and int(msg.get("episode_id", -1)) == target_episode:
                    if not reset_done_seen:
                        # Defensive: release should follow reset_done, but recover if a
                        # broadcast packet arrived out of order by asking again.
                        continue
                    self.step_count = 0
                    self.episode_reward = 0.0
                    self._reset_episode_counters()
                    self.start_wall = time.perf_counter()
                    self.last_outcome = None
                    obs = self.read_observation()
                    self.prev_target_signal = float(max(obs[0], obs[2], obs[3]))
                    return obs, {
                        "episode_id": self.episode_id,
                        "reset_integrity": "ok",
                    }
                if mtype == "shutdown":
                    raise RuntimeError("Supervisor requested shutdown while waiting for reset")

            waited_steps += 1
            if waited_steps >= max_wait_steps:
                raise RuntimeError(
                    f"Timed out waiting for reset_done for episode {target_episode} "
                    f"after {waited_steps} Webots steps. Check supervisor communication."
                )
        raise RuntimeError("Webots simulation ended while waiting for reset")

    def _curriculum_attack_multiplier(self) -> float:
        table = self.cfg.get("reward_design", {}).get("curriculum_attack_multipliers", {}) or {}
        return float(table.get(self.curriculum_level, 1.0))

    def _dense_reward(self, obs: np.ndarray, direction: str, speed_label: str, left_cmd: float, right_cmd: float) -> float:
        """v1.12 attack-learning reward with physical motion semantics.

        Key design changes from v1.10:
        * no unconditional reward merely for seeing a frontal target;
        * frontal visibility is rewarded only when the robot commits forward;
        * not attacking a centered target is mildly penalized;
        * real first contact and sustained forward contact receive explicit reward;
        * Supervisor center-distance and opponent-radius deltas provide precise,
          training-only approach/push shaping while the policy observation stays
          hardware-reproducible;
        * terminal win/loss remains much larger than any single dense event.
        """
        ec = self.cfg["environment"]
        front, back, left, right, line_l, line_r = [float(x) for x in obs]
        reward = float(ec["time_penalty"])
        mult = self._curriculum_attack_multiplier() if self.phase == "train" else 1.0
        motion = motion_flags(direction, left_cmd, right_cmd)
        is_forward_translation = motion["translational_forward"]
        is_straight_forward = motion["straight_forward"]
        is_curved_forward = motion["curved_forward"]

        edge_risk = line_l > 0.5 or line_r > 0.5
        target_signal = max(front, left, right)
        target_detected = (target_signal >= self.target_detect_floor) or (back >= 0.5)
        if target_detected:
            self.target_detected_steps += 1

        # Search when the target is not visible. Rotation is the safer default;
        # controlled forward search remains mildly useful but not dominant.
        if not target_detected and not edge_risk:
            if is_forward_translation:
                reward += float(ec.get("safe_search_forward_reward", 0.0))
                self.search_forward_steps += 1
            elif direction in {"spin_left", "spin_right"}:
                reward += float(ec.get("scan_reward", 0.0))
                self.scan_steps += 1

        # Privileged center-distance shaping (training reward only, not observation).
        if (
            target_detected
            and self.latest_distance_m is not None
            and self.prev_distance_m is not None
        ):
            ddelta = self.prev_distance_m - self.latest_distance_m  # positive = approach
            if ddelta > 0 and direction in {"forward", "left", "right"}:
                reward += float(ec.get("approach_distance_reward_scale", 0.0)) * ddelta
                self.approach_progress_steps += 1
            elif ddelta < 0:
                reward += float(ec.get("retreat_distance_penalty_scale", 0.0)) * abs(ddelta)
                self.retreat_from_target_steps += 1

        # Frontal target: attack commitment is the desired local action.
        if front >= self.front_attack_threshold:
            self.engagement_steps += 1
            if is_forward_translation:
                # Straight and curved forward commands are both valid attack motion.
                reward += mult * float(ec.get("attack_reward", 0.0))
                if speed_label == "high":
                    reward += mult * float(ec.get("attack_high_speed_bonus", 0.0))
                elif speed_label == "medium":
                    reward += mult * float(ec.get("attack_medium_speed_bonus", 0.0))
                self.attack_steps += 1
            elif not edge_risk:
                reward += mult * float(ec.get("visible_target_nonforward_penalty", 0.0))
                self.front_visible_nonforward_steps += 1
                if direction == "backward":
                    reward += mult * float(ec.get("backward_from_visible_target_penalty", 0.0))
                    self.front_visible_backward_steps += 1
                elif direction in {"spin_left", "spin_right"}:
                    reward += mult * float(ec.get("front_spin_penalty", 0.0))

        # Side/rear detections still teach orientation, but with less reward than attack.
        if left >= self.side_detect_threshold and left > right and direction in {"left", "spin_left"}:
            reward += float(ec.get("correct_turn_reward", 0.0))
            self.turn_toward_steps += 1
        if right >= self.side_detect_threshold and right > left and direction in {"right", "spin_right"}:
            reward += float(ec.get("correct_turn_reward", 0.0))
            self.turn_toward_steps += 1
        if back >= 0.5 and direction in {"spin_left", "spin_right"}:
            reward += float(ec.get("rear_turn_reward", 0.0))
            self.turn_toward_steps += 1

        # Real contact from Supervisor telemetry: this is the core attack-learning cue.
        if self.latest_contact:
            self.contact_steps_local += 1
            if self.contact_started_this_step:
                reward += mult * float(ec.get("first_contact_reward", 0.0))
                self.first_contact_events += 1
            if is_forward_translation:
                reward += mult * float(ec.get("sustained_contact_forward_reward", 0.0))
                self.contact_forward_steps += 1
            elif not edge_risk:
                reward += float(ec.get("contact_nonforward_penalty", 0.0))

            if (
                is_forward_translation
                and self.latest_opponent_radius_m is not None
                and self.prev_opponent_radius_m is not None
            ):
                radial_delta = self.latest_opponent_radius_m - self.prev_opponent_radius_m
                if radial_delta > 0:
                    reward += float(ec.get("push_outward_reward_scale", 0.0)) * radial_delta
                    self.push_outward_progress_steps += 1

        if edge_risk:
            reward += float(ec["edge_risk_penalty"])
            self.edge_risk_steps += 1
        elif direction == "backward":
            reward += float(ec.get("backward_without_edge_penalty", 0.0))
            self.backward_without_edge_steps += 1

        if is_forward_translation:
            self.forward_steps += 1
        if is_straight_forward:
            self.straight_forward_steps += 1
        if is_curved_forward:
            self.curved_forward_steps += 1
        if speed_label == "high":
            self.high_speed_steps += 1

        self.prev_target_signal = target_signal
        if self.latest_distance_m is not None:
            self.prev_distance_m = self.latest_distance_m
        if self.latest_opponent_radius_m is not None:
            self.prev_opponent_radius_m = self.latest_opponent_radius_m
        return reward

    def _terminal_reward_components(self, outcome: Dict[str, Any]) -> Tuple[float, float, float]:
        ec = self.cfg["environment"]
        winner = outcome.get("winner")
        reason = outcome.get("reason", "")
        cause = outcome.get("ring_out_cause", "")
        if not bool(outcome.get("valid_episode", True)):
            # Reset-integrity artifacts are diagnostic only and must not shape the policy.
            return 0.0, 0.0, 0.0
        loser = outcome.get("loser")

        if winner is None:
            base = float(ec["timeout_reward"] if reason == "timeout" else 0.0)
            return base, base, 0.0

        if int(winner) == self.agent_id:
            base = float(ec["terminal_win_reward"])
            if cause == "contact_assisted":
                shaping = float(ec.get("forced_ringout_bonus", 0.0))
            else:
                shaping = float(ec.get("unforced_win_bonus", 0.0))
            return base + shaping, base, shaping

        base = float(ec["terminal_loss_reward"])
        shaping = 0.0
        if loser is not None and int(loser) == self.agent_id and cause == "unforced_self_out":
            shaping += float(ec.get("unforced_self_out_penalty", 0.0))
        return base + shaping, base, shaping

    def _episode_metrics(
        self,
        outcome: Dict[str, Any],
        current_reward: float,
        official_terminal_reward: float,
        causal_shaping_reward: float,
    ) -> Dict[str, Any]:
        steps = max(1, self.step_count)
        winner = outcome.get("winner")
        valid_episode = int(bool(outcome.get("valid_episode", True)))
        if valid_episode:
            win = int(winner is not None and int(winner) == self.agent_id)
            loss = int(winner is not None and int(winner) != self.agent_id)
            draw = int(winner is None)
        else:
            win = loss = draw = 0
        translational_forward_ratio = self.forward_steps / steps
        # Backward-compatible alias: from v1.12 onward, forward_ratio means
        # physically forward-translational motion (both wheels positive).
        forward_ratio = translational_forward_ratio
        straight_forward_ratio = self.straight_forward_steps / steps
        curved_forward_ratio = self.curved_forward_steps / steps
        high_speed_ratio = self.high_speed_steps / steps
        attack_ratio = self.attack_steps / steps
        engagement_ratio = self.engagement_steps / steps
        edge_risk_ratio = self.edge_risk_steps / steps
        turn_toward_ratio = self.turn_toward_steps / steps
        search_forward_ratio = self.search_forward_steps / steps
        scan_ratio = self.scan_steps / steps
        target_detected_ratio = self.target_detected_steps / steps
        approach_progress_ratio = self.approach_progress_steps / steps
        retreat_from_target_ratio = self.retreat_from_target_steps / steps
        backward_without_edge_ratio = self.backward_without_edge_steps / steps
        front_visible_nonforward_ratio = self.front_visible_nonforward_steps / steps
        contact_ratio = self.contact_steps_local / steps
        contact_forward_ratio = self.contact_forward_steps / steps
        push_outward_progress_ratio = self.push_outward_progress_steps / steps

        aggression_index = (
            0.22 * forward_ratio
            + 0.13 * high_speed_ratio
            + 0.22 * engagement_ratio
            + 0.20 * attack_ratio
            + 0.13 * contact_ratio
            + 0.10 * target_detected_ratio
        )
        duration_s = steps * self.timestep / 1000.0
        outcome_label = "invalid" if not valid_episode else ("draw" if draw else ("win" if win else "loss"))
        contact = outcome.get("ring_out_contact_point")
        if isinstance(contact, (list, tuple)) and len(contact) >= 3:
            contact_x, contact_y, contact_z = [float(contact[i]) for i in range(3)]
        else:
            contact_x = contact_y = contact_z = math.nan

        cause = outcome.get("ring_out_cause", "")
        reset_integrity_failure = int(not bool(valid_episode))
        forced_win = int(valid_episode and win and cause == "contact_assisted")
        unforced_win = int(valid_episode and win and cause == "unforced_self_out")
        age_steps = outcome.get("last_robot_contact_age_steps")
        age_s = outcome.get("last_robot_contact_age_s")

        dominant_idx = int(np.argmax(self.action_counts)) if self.action_counts.sum() else -1
        dominant_name = action_name(dominant_idx) if dominant_idx >= 0 else "none"
        probs = self.action_counts.astype(float) / max(1, int(self.action_counts.sum()))
        nz = probs[probs > 0]
        empirical_action_entropy = float(-(nz * np.log(nz)).sum()) if len(nz) else 0.0

        row = {
            "experiment_id": self.cfg.get("experiment", {}).get("id", ""),
            "stage": self.cfg.get("experiment", {}).get("stage", ""),
            "seed": self.cfg.get("experiment", {}).get("seed", ""),
            "agent_id": self.agent_id,
            "phase": self.phase,
            "curriculum_level": self.curriculum_level,
            "curriculum_progress": self.curriculum_progress,
            "episode_index": self.episode_index,
            "episode_id": self.episode_id,
            "total_env_steps": self.total_env_steps,
            "episode_steps": steps,
            "duration_s": duration_s,
            "episode_reward": self.episode_reward + current_reward,
            "official_terminal_reward": official_terminal_reward,
            "causal_shaping_reward": causal_shaping_reward,
            "winner": "invalid" if not valid_episode else ("draw" if winner is None else int(winner)),
            "outcome": outcome_label,
            "loser": outcome.get("loser", ""),
            "reason": outcome.get("reason", ""),
            "valid_episode": valid_episode,
            "reset_integrity_failure": reset_integrity_failure,
            "termination_criterion": outcome.get("termination_criterion", ""),
            "ring_out_cause": cause,
            "forced_win": forced_win,
            "unforced_win": unforced_win,
            "last_robot_contact_age_steps": "" if age_steps is None else age_steps,
            "last_robot_contact_age_s": math.nan if age_s is None else float(age_s),
            "robot_contact_steps": int(outcome.get("robot_contact_steps", 0) or 0),
            "ring_out_contact_x_m": contact_x,
            "ring_out_contact_y_m": contact_y,
            "ring_out_contact_z_m": contact_z,
            "win": win,
            "loss": loss,
            "draw": draw,
            "forward_ratio": forward_ratio,
            "translational_forward_ratio": translational_forward_ratio,
            "straight_forward_ratio": straight_forward_ratio,
            "curved_forward_ratio": curved_forward_ratio,
            "high_speed_ratio": high_speed_ratio,
            "attack_ratio": attack_ratio,
            "engagement_ratio": engagement_ratio,
            "edge_risk_ratio": edge_risk_ratio,
            "turn_toward_ratio": turn_toward_ratio,
            "search_forward_ratio": search_forward_ratio,
            "scan_ratio": scan_ratio,
            "target_detected_ratio": target_detected_ratio,
            "approach_progress_ratio": approach_progress_ratio,
            "retreat_from_target_ratio": retreat_from_target_ratio,
            "backward_without_edge_ratio": backward_without_edge_ratio,
            "front_visible_nonforward_ratio": front_visible_nonforward_ratio,
            "contact_ratio": contact_ratio,
            "contact_forward_ratio": contact_forward_ratio,
            "first_contact_events": self.first_contact_events,
            "push_outward_progress_ratio": push_outward_progress_ratio,
            "dominant_action_id": dominant_idx,
            "dominant_action_name": dominant_name,
            "empirical_action_entropy": empirical_action_entropy,
            "aggression_index": aggression_index,
            "wall_time_s": (time.perf_counter() - self.start_wall) if self.start_wall else math.nan,
        }
        for idx, count in enumerate(self.action_counts.tolist()):
            row[f"action_{idx}_{action_name(idx)}_ratio"] = float(count) / steps
        return row

    def _write_episode(self, metrics: Dict[str, Any]) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            if not self._csv_header_written:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerow(metrics)

    def step(self, action):
        _idx, direction, speed_label, left_cmd, right_cmd = self._apply_action(action)
        if self.robot.step(self.timestep) == -1:
            raise RuntimeError("Webots simulation ended")

        self.step_count += 1
        self.total_env_steps += 1
        obs = self.read_observation()
        # Process Supervisor telemetry before reward computation so contact/distance
        # shaping corresponds to this exact physical simulation step.
        outcome = self._process_messages()
        reward = self._dense_reward(obs, direction, speed_label, left_cmd, right_cmd)

        terminated = False
        truncated = False
        info: Dict[str, Any] = {"episode_id": self.episode_id}

        if outcome is not None:
            terminal, official_base, causal = self._terminal_reward_components(outcome)
            reward += terminal
            self.last_outcome = outcome
            if not bool(outcome.get("valid_episode", True)):
                terminated = False
                truncated = True
            else:
                terminated = outcome.get("reason") != "timeout"
                truncated = outcome.get("reason") == "timeout"
            metrics = self._episode_metrics(outcome, reward, official_base, causal)
            self._write_episode(metrics)
            info.update(metrics)
            info["episode_metrics"] = metrics
            self.episode_index += 1

        self.episode_reward += reward
        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self):
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
