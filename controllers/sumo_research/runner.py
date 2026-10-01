from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from common import drain_json, get_agent_cfg, resolve_run_dir, runtime_manifest, save_json, seed_everything, send_json
from env import ACTION_TABLE, WebotsSumoEnv, action_speed, max_drive_speed, motor_command_for_action
from plots import make_evaluation_plots, make_training_plots
from rl_factory import create_model, load_model, normalized_name, predict_action


class ProgressCallbackFactory:
    @staticmethod
    def build(run_dir: Path, checkpoint_every: int):
        from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

        class StatusCallback(BaseCallback):
            def __init__(self, status_path: Path, verbose=0):
                super().__init__(verbose)
                self.status_path = status_path
                self.started = time.time()
                self.last_write = -1

            def _on_step(self) -> bool:
                if self.num_timesteps - self.last_write >= 10000:
                    payload = {
                        "status": "training",
                        "num_timesteps": int(self.num_timesteps),
                        "elapsed_s": time.time() - self.started,
                    }
                    save_json(self.status_path, payload)
                    self.last_write = self.num_timesteps
                return True

        callbacks = [StatusCallback(run_dir / "status.json")]
        if checkpoint_every > 0:
            callbacks.append(CheckpointCallback(
                save_freq=max(1, checkpoint_every),
                save_path=str(run_dir / "checkpoints"),
                name_prefix="checkpoint",
                save_replay_buffer=False,
                save_vecnormalize=False,
            ))
        return CallbackList(callbacks)


def run_training(robot, cfg: Dict[str, Any], agent_id: int) -> None:
    agent_cfg = get_agent_cfg(cfg, agent_id)
    algo = normalized_name(agent_cfg.get("algorithm", "PPO"))
    seed = int(cfg.get("experiment", {}).get("seed", 0)) + agent_id * 10000
    deterministic_torch = bool(cfg.get("training", {}).get("deterministic_torch", False))
    seed_everything(seed, deterministic_torch)

    run_dir = resolve_run_dir(cfg, agent_id)
    save_json(run_dir / "runtime_manifest.json", runtime_manifest(cfg, agent_id, algo))
    with (run_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    base_env = WebotsSumoEnv(robot, cfg, agent_id, phase="train")
    monitor_path = run_dir / "monitor"
    monitored = Monitor(
        base_env,
        filename=str(monitor_path),
        info_keywords=("win", "loss", "draw", "forced_win", "unforced_win", "aggression_index",
                       "engagement_ratio", "attack_ratio", "target_detected_ratio",
                       "search_forward_ratio", "approach_progress_ratio", "edge_risk_ratio",
                       "contact_ratio", "contact_forward_ratio", "first_contact_events",
                       "front_visible_nonforward_ratio", "push_outward_progress_ratio",
                       "translational_forward_ratio", "straight_forward_ratio",
                       "curved_forward_ratio", "valid_episode", "reset_integrity_failure",
                       "dominant_action_id", "empirical_action_entropy"),
    )
    vec_env = DummyVecEnv([lambda: monitored])

    tb_dir = str(run_dir / "tensorboard") if cfg.get("training", {}).get("tensorboard", True) else None
    resume = agent_cfg.get("resume_model")
    if resume:
        model = load_model(algo, resume, env=vec_env, tensorboard_log=tb_dir)
        if algo == "DQN":
            replay_path = Path(resume).resolve().parent / "replay_buffer.pkl"
            if replay_path.exists():
                model.load_replay_buffer(str(replay_path))
                print(f"[training] restored DQN replay buffer: {replay_path}")
        reset_num_timesteps = False
    else:
        model = create_model(algo, vec_env, cfg, seed, tb_dir)
        reset_num_timesteps = True

    total_timesteps = int(cfg.get("training", {}).get("total_timesteps", 500000))
    checkpoint_every = int(cfg.get("training", {}).get("checkpoint_every_steps", 50000))
    callback = ProgressCallbackFactory.build(run_dir, checkpoint_every)

    started = time.time()
    save_json(run_dir / "status.json", {"status": "starting", "algorithm": algo, "seed": seed})
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name=f"{algo}_agent{agent_id}",
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=False,
        )
        model_path = run_dir / "model_final"
        model.save(str(model_path))
        if algo == "DQN":
            replay_path = run_dir / "replay_buffer.pkl"
            model.save_replay_buffer(str(replay_path))
            print(f"[training] saved DQN replay buffer: {replay_path}")
        save_json(run_dir / "status.json", {
            "status": "complete",
            "algorithm": algo,
            "seed": seed,
            "elapsed_s": time.time() - started,
            "model": str(model_path) + ".zip",
        })
        make_training_plots(base_env.csv_path, run_dir / "figures")
        send_json(base_env.emitter, {
            "type": "agent_done",
            "agent_id": agent_id,
            "purpose": "training",
            "model": str(model_path) + ".zip",
        })
    except Exception as exc:
        save_json(run_dir / "status.json", {"status": "failed", "error": repr(exc)})
        send_json(base_env.emitter, {"type": "agent_failed", "agent_id": agent_id, "error": repr(exc)})
        raise
    finally:
        base_env.close()

    # Keep this synchronized controller alive until the supervisor closes Webots.
    while robot.step(int(robot.getBasicTimeStep())) != -1:
        pass


def _append_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def run_evaluation(robot, cfg: Dict[str, Any], agent_id: int) -> None:
    agent_cfg = get_agent_cfg(cfg, agent_id)
    algo = normalized_name(agent_cfg.get("algorithm", "PPO"))
    model_path = agent_cfg.get("model_path") or agent_cfg.get("resume_model")
    if not model_path:
        raise ValueError(f"Agent {agent_id} evaluation requires model_path")

    run_dir = resolve_run_dir(cfg, agent_id)
    with (run_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    env = WebotsSumoEnv(robot, cfg, agent_id, phase="eval")
    print(f"[evaluation] agent={agent_id} loading {algo} model: {model_path}")
    model = load_model(algo, model_path, env=None)
    print(f"[evaluation] agent={agent_id} model loaded; requesting episode reset")
    deterministic = bool(cfg.get("evaluation", {}).get("deterministic", True))
    n_episodes = int(cfg.get("evaluation", {}).get("episodes", 100))
    out_csv = run_dir / "evaluation.csv"
    if out_csv.exists():
        out_csv.unlink()

    obs, _ = env.reset()
    state = None
    episode_start = np.ones((1,), dtype=bool)
    completed = 0
    while completed < n_episodes:
        action, state = predict_action(model, algo, obs, deterministic, state=state, episode_start=episode_start)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_start = np.asarray([done], dtype=bool)
        if done:
            metrics = dict(info.get("episode_metrics", {}))
            metrics["algorithm"] = algo
            metrics["model_path"] = str(model_path)
            metrics["opponent_mode"] = get_agent_cfg(cfg, 1 - agent_id).get("mode", "")
            metrics["opponent_profile"] = get_agent_cfg(cfg, 1 - agent_id).get("scripted_profile", "")
            valid_episode = bool(int(metrics.get("valid_episode", 1) or 0))
            if valid_episode:
                metrics["evaluation_episode"] = completed
                _append_csv(out_csv, metrics)
                completed += 1
                print(
                    f"[evaluation] agent={agent_id} episode {completed}/{n_episodes} "
                    f"outcome={metrics.get('outcome')} winner={metrics.get('winner')} "
                    f"reason={metrics.get('reason')} criterion={metrics.get('termination_criterion')} "
                    f"cause={metrics.get('ring_out_cause')} steps={metrics.get('episode_steps')} "
                    f"reward={metrics.get('episode_reward', float('nan')):.3f}"
                )
            else:
                invalid_csv = run_dir / "evaluation_invalid_reset_artifacts.csv"
                metrics["evaluation_episode"] = "discarded"
                _append_csv(invalid_csv, metrics)
                print(
                    f"[evaluation] DISCARD reset-integrity artifact episode_id={metrics.get('episode_id')} "
                    f"steps={metrics.get('episode_steps')} reason={metrics.get('reason')} -- retrying"
                )
            if completed < n_episodes:
                obs, _ = env.reset()
                state = None
                episode_start = np.ones((1,), dtype=bool)

    print(f"[evaluation] agent={agent_id} completed {completed} episodes; saving plots")
    make_evaluation_plots(out_csv, run_dir / "figures")
    save_json(run_dir / "status.json", {"status": "evaluation_complete", "episodes": completed})
    send_json(env.emitter, {"type": "agent_done", "agent_id": agent_id, "purpose": "evaluation"})
    env.close()
    while robot.step(int(robot.getBasicTimeStep())) != -1:
        pass


def _set_velocity(left_motor, right_motor, left: float, right: float, limit: float = 22.0):
    limit = abs(float(limit))
    left_motor.setVelocity(float(np.clip(left, -limit, limit)))
    right_motor.setVelocity(float(np.clip(right, -limit, limit)))


def run_static(robot, cfg: Dict[str, Any], agent_id: int) -> None:
    timestep = int(robot.getBasicTimeStep())
    left_motor = robot.getDevice("left wheel motor")
    right_motor = robot.getDevice("right wheel motor")
    left_motor.setPosition(float("inf")); right_motor.setPosition(float("inf"))
    _set_velocity(left_motor, right_motor, 0, 0)
    receiver = robot.getDevice("receiver"); receiver.enable(timestep)
    while robot.step(timestep) != -1:
        _set_velocity(left_motor, right_motor, 0, 0)
        drain_json(receiver)


def _norm(value: float, sensor_max: float) -> float:
    return float(np.clip(value / sensor_max, 0, 1))


def _edge_risk(value: float, risk_low: float, risk_high: float) -> bool:
    """Return True when a downward edge sensor no longer sees normal Dohyo surface."""
    return bool(value < risk_low or value > risk_high)


def _classify_edge(gl: float, gr: float, glr: float, grr: float,
                   risk_low: float, risk_high: float) -> Dict[str, bool]:
    """Classify the four edge sensors into front/rear and left/right risk sectors.

    v1.9 treated every edge event identically and always reversed. That is unsafe
    when a *rear* sensor is over the boundary because reversing moves farther out.
    v1.10 keeps all robot/arena physics unchanged and fixes only this controller
    logic for scripted benchmark opponents.
    """
    fl = _edge_risk(gl, risk_low, risk_high)
    fr = _edge_risk(gr, risk_low, risk_high)
    rl = _edge_risk(glr, risk_low, risk_high)
    rr = _edge_risk(grr, risk_low, risk_high)
    return {
        "fl": fl, "fr": fr, "rl": rl, "rr": rr,
        "front": fl or fr,
        "rear": rl or rr,
        "left": fl or rl,
        "right": fr or rr,
        "any": fl or fr or rl or rr,
    }


def _choose_edge_escape(edge: Dict[str, bool], previous_turn_left: bool):
    """Choose translational escape direction and an inward turn direction.

    Returns `(escape_direction, turn_left)` where escape_direction is:
      -1 = reverse (front edge detected), +1 = forward (rear edge detected),
       0 = rotate in place (ambiguous front+rear edge state).
    """
    front_count = int(edge["fl"]) + int(edge["fr"])
    rear_count = int(edge["rl"]) + int(edge["rr"])
    if edge["front"] and not edge["rear"]:
        escape_direction = -1
    elif edge["rear"] and not edge["front"]:
        escape_direction = +1
    elif front_count > rear_count:
        escape_direction = -1
    elif rear_count > front_count:
        escape_direction = +1
    else:
        escape_direction = 0

    # Turn away from the side that detected the boundary. If both sides are
    # ambiguous, alternate deterministically from the previous escape event.
    if edge["left"] and not edge["right"]:
        turn_left = False
    elif edge["right"] and not edge["left"]:
        turn_left = True
    else:
        turn_left = not previous_turn_left
    return escape_direction, turn_left


def run_scripted(robot, cfg: Dict[str, Any], agent_id: int) -> None:
    """Run a deterministic benchmark opponent with a v1.10 edge-safety guard.

    The normal search/turn/attack policy is intentionally unchanged from v1.9.
    Only the edge-recovery layer is strengthened so scripted/aggressive opponents
    do not create artificial RL wins by driving themselves out of the Dohyo.
    """
    timestep = int(robot.getBasicTimeStep())
    acfg = get_agent_cfg(cfg, agent_id)
    profile_name = acfg.get("scripted_profile", "standard")
    p = cfg.get("scripted_opponents", {}).get(profile_name, cfg.get("scripted_opponents", {}).get("standard", {}))
    drive_limit = max_drive_speed(cfg)
    sensor_max = float(cfg["environment"]["sensor_max"])
    front_thr = float(cfg["environment"]["front_engage_threshold"])
    side_thr = float(cfg["environment"]["side_detect_threshold"])
    risk_low = float(cfg["environment"]["line_risk_low"])
    risk_high = float(cfg["environment"]["line_risk_high"])

    left_motor = robot.getDevice("left wheel motor"); right_motor = robot.getDevice("right wheel motor")
    left_motor.setPosition(float("inf")); right_motor.setPosition(float("inf"))
    _set_velocity(left_motor, right_motor, 0, 0, drive_limit)
    reset_hold = True
    front = robot.getDevice("front_distance_sensor"); left_s = robot.getDevice("left_side_sensor"); right_s = robot.getDevice("right_side_sensor")
    line_l = robot.getDevice("left_ground_sensor"); line_r = robot.getDevice("right_ground_sensor")
    line_lr = robot.getDevice("left_rear_ground_sensor"); line_rr = robot.getDevice("right_rear_ground_sensor")
    for sensor in [front, left_s, right_s, line_l, line_r, line_lr, line_rr]:
        sensor.enable(timestep)
    receiver = robot.getDevice("receiver"); receiver.enable(timestep)

    escape_countdown = 0
    turn_countdown = 0
    guard_countdown = 0
    escape_direction = -1
    turn_left = True
    edge_active = False
    edge_events = 0

    escape_steps = int(p.get("edge_escape_steps", p.get("edge_backup_steps", 90)))
    edge_turn_steps = int(p.get("edge_turn_steps", 100))
    guard_steps = int(p.get("post_edge_guard_steps", 70))
    emergency_speed = float(p.get("edge_escape_speed", p.get("reverse_speed", 11.0)))
    edge_turn_speed = float(p.get("edge_turn_speed", p.get("turn_speed", 11.0)))

    print(
        f"[scripted] agent={agent_id} profile={profile_name} "
        f"edge_guard=directional_v1.10 escape_steps={escape_steps} "
        f"turn_steps={edge_turn_steps} guard_steps={guard_steps}"
    )

    def command_escape() -> None:
        if escape_direction < 0:
            _set_velocity(left_motor, right_motor, -emergency_speed, -emergency_speed, drive_limit)
        elif escape_direction > 0:
            _set_velocity(left_motor, right_motor, emergency_speed, emergency_speed, drive_limit)
        else:
            # Ambiguous front+rear risk: translate no farther; rotate toward the
            # previously selected safe side until one pair of sensors clears.
            _set_velocity(
                left_motor, right_motor,
                -edge_turn_speed if turn_left else edge_turn_speed,
                edge_turn_speed if turn_left else -edge_turn_speed,
                drive_limit,
            )

    def command_turn(speed: float) -> None:
        _set_velocity(
            left_motor, right_motor,
            -speed if turn_left else speed,
            speed if turn_left else -speed,
            drive_limit,
        )

    while robot.step(timestep) != -1:
        msgs = drain_json(receiver)
        if any(m.get("type") == "reset_prepare" for m in msgs):
            reset_hold = True
            _set_velocity(left_motor, right_motor, 0, 0, drive_limit)
        if any(m.get("type") == "reset_done" for m in msgs):
            # Pose is verified, but remain frozen until the learner acknowledges it
            # and the supervisor broadcasts reset_release.
            reset_hold = True
            escape_countdown = 0
            turn_countdown = 0
            guard_countdown = 0
            edge_active = False
            _set_velocity(left_motor, right_motor, 0, 0, drive_limit)
        if any(m.get("type") == "reset_release" for m in msgs):
            reset_hold = False
            escape_countdown = 0
            turn_countdown = 0
            guard_countdown = 0
            edge_active = False
        if reset_hold:
            _set_velocity(left_motor, right_motor, 0, 0, drive_limit)
            continue

        f = _norm(front.getValue(), sensor_max)
        l = _norm(left_s.getValue(), sensor_max)
        r = _norm(right_s.getValue(), sensor_max)
        gl = _norm(line_l.getValue(), sensor_max)
        gr = _norm(line_r.getValue(), sensor_max)
        glr = _norm(line_lr.getValue(), sensor_max)
        grr = _norm(line_rr.getValue(), sensor_max)
        edge = _classify_edge(gl, gr, glr, grr, risk_low, risk_high)

        # Edge sensing has absolute priority over attack/search. Start the escape
        # in the SAME controller step as detection, rather than allowing one more
        # step of the previous forward command.
        if edge["any"]:
            if not edge_active:
                escape_direction, turn_left = _choose_edge_escape(edge, turn_left)
                edge_events += 1
            else:
                # If the opposite end reaches the boundary while translating,
                # stop translating rather than oscillating across the Tawara.
                if (escape_direction < 0 and edge["rear"]) or (escape_direction > 0 and edge["front"]):
                    _, turn_left = _choose_edge_escape(edge, turn_left)
                    escape_direction = 0
                elif escape_direction == 0 and edge["front"] != edge["rear"]:
                    escape_direction, turn_left = _choose_edge_escape(edge, turn_left)
            edge_active = True
            escape_countdown = max(escape_countdown, escape_steps)
            turn_countdown = 0
            guard_countdown = max(guard_countdown, guard_steps)
            command_escape()
            continue

        if edge_active:
            # Boundary has just cleared. Preserve the chosen inward escape for a
            # short fixed distance before turning, giving the 3 kg robot enough
            # braking margin despite attack momentum.
            edge_active = False

        if escape_countdown > 0:
            command_escape()
            escape_countdown -= 1
            if escape_countdown == 0:
                turn_countdown = edge_turn_steps
            continue

        if turn_countdown > 0:
            command_turn(edge_turn_speed)
            turn_countdown -= 1
            continue

        if guard_countdown > 0:
            # Post-edge guard deliberately rotates instead of translating. This
            # prevents an immediate attack command from sending the opponent back
            # over the same boundary before all four sensors have stabilized.
            command_turn(min(edge_turn_speed, float(p.get("turn_speed", edge_turn_speed))))
            guard_countdown -= 1
            continue

        attack = float(p.get("attack_speed", 85))
        turn = float(p.get("turn_speed", 70))
        search = float(p.get("search_speed", 55))
        if f >= front_thr:
            _set_velocity(left_motor, right_motor, attack, attack, drive_limit)
        elif l >= side_thr and l > r:
            _set_velocity(left_motor, right_motor, -0.6 * turn, turn, drive_limit)
        elif r >= side_thr and r > l:
            _set_velocity(left_motor, right_motor, turn, -0.6 * turn, drive_limit)
        else:
            # Pure in-place search is intentionally translation-free, minimizing
            # random edge drift when the RL robot is not detected.
            _set_velocity(left_motor, right_motor, search, -search, drive_limit)

def run_frozen(robot, cfg: Dict[str, Any], agent_id: int) -> None:
    acfg = get_agent_cfg(cfg, agent_id)
    algo = normalized_name(acfg.get("algorithm", "PPO"))
    model_path = acfg.get("model_path") or acfg.get("resume_model")
    if not model_path:
        raise ValueError(f"Frozen agent {agent_id} requires model_path")
    model = load_model(algo, model_path)
    deterministic = bool(acfg.get("deterministic_eval", True))

    timestep = int(robot.getBasicTimeStep())
    drive_limit = max_drive_speed(cfg)
    sensor_max = float(cfg["environment"]["sensor_max"])
    back_thr = float(cfg["environment"].get("back_detect_threshold", 0.15))
    risk_low = float(cfg["environment"]["line_risk_low"])
    risk_high = float(cfg["environment"]["line_risk_high"])
    left_motor = robot.getDevice("left wheel motor"); right_motor = robot.getDevice("right wheel motor")
    left_motor.setPosition(float("inf")); right_motor.setPosition(float("inf"))
    front = robot.getDevice("front_distance_sensor"); back = robot.getDevice("back_sensor")
    left_s = robot.getDevice("left_side_sensor"); right_s = robot.getDevice("right_side_sensor")
    line_l = robot.getDevice("left_ground_sensor"); line_r = robot.getDevice("right_ground_sensor")
    line_lr = robot.getDevice("left_rear_ground_sensor"); line_rr = robot.getDevice("right_rear_ground_sensor")
    for s in [front, back, left_s, right_s, line_l, line_r, line_lr, line_rr]: s.enable(timestep)
    receiver = robot.getDevice("receiver"); receiver.enable(timestep)
    state = None
    episode_start = np.ones((1,), dtype=bool)
    reset_hold = True
    _set_velocity(left_motor, right_motor, 0, 0, drive_limit)

    while robot.step(timestep) != -1:
        msgs = drain_json(receiver)
        if any(m.get("type") == "reset_prepare" for m in msgs):
            reset_hold = True
            _set_velocity(left_motor, right_motor, 0, 0, drive_limit)
        if any(m.get("type") == "reset_done" for m in msgs):
            reset_hold = True
            state = None
            episode_start = np.ones((1,), dtype=bool)
            _set_velocity(left_motor, right_motor, 0, 0, drive_limit)
        if any(m.get("type") == "reset_release" for m in msgs):
            reset_hold = False
            state = None
            episode_start = np.ones((1,), dtype=bool)
        if reset_hold:
            _set_velocity(left_motor, right_motor, 0, 0, drive_limit)
            continue
        front_v = _norm(front.getValue(), sensor_max)
        back_v = _norm(back.getValue(), sensor_max)
        left_v = _norm(left_s.getValue(), sensor_max)
        right_v = _norm(right_s.getValue(), sensor_max)
        line_l_v = _norm(line_l.getValue(), sensor_max)
        line_r_v = _norm(line_r.getValue(), sensor_max)
        line_lr_v = _norm(line_lr.getValue(), sensor_max)
        line_rr_v = _norm(line_rr.getValue(), sensor_max)
        obs = np.asarray([
            front_v,
            float(back_v >= back_thr),
            left_v,
            right_v,
            float(
                line_l_v < risk_low or line_l_v > risk_high
                or line_lr_v < risk_low or line_lr_v > risk_high
            ),
            float(
                line_r_v < risk_low or line_r_v > risk_high
                or line_rr_v < risk_low or line_rr_v > risk_high
            ),
        ], dtype=np.float32)
        action, state = predict_action(model, algo, obs, deterministic, state=state, episode_start=episode_start)
        episode_start = np.zeros((1,), dtype=bool)
        idx = int(np.asarray(action).reshape(-1)[0])
        _direction, _speed_label, left, right = motor_command_for_action(cfg, idx)
        _set_velocity(left_motor, right_motor, left, right, drive_limit)
