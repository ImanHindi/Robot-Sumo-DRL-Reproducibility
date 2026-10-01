from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml
from controller import Supervisor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "active_experiment.yaml"


def send_json(emitter, payload):
    emitter.send(json.dumps(payload, separators=(",", ":")).encode("utf-8"))


def read_messages(receiver):
    out = []
    while receiver.getQueueLength() > 0:
        text = receiver.getString()
        receiver.nextPacket()
        try:
            out.append(json.loads(text))
        except Exception:
            pass
    return out


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def yaw_to_face(source_xy, target_xy):
    dx = target_xy[0] - source_xy[0]
    dy = target_xy[1] - source_xy[1]
    # Robot local forward direction is -Y.
    return math.atan2(dx, -dy)


def _curriculum_level(cfg, training_steps: int):
    curriculum = cfg.get("curriculum", {})
    profile_name = str(curriculum.get("profile", ""))
    profile = curriculum.get(profile_name) if curriculum.get("enabled", False) else None
    if not isinstance(profile, dict):
        return None, "legacy_randomized", 1.0
    levels = list(profile.get("levels", []))
    breaks = list(profile.get("breaks", []))
    if not levels:
        return None, profile_name or "legacy_randomized", 1.0
    total = max(1, int(cfg.get("training", {}).get("total_timesteps", 1)))
    progress = min(1.0, max(0.0, float(training_steps) / float(total)))
    idx = len(levels) - 1
    for i, boundary in enumerate(breaks[:len(levels)]):
        if progress < float(boundary) or i == len(levels) - 1:
            idx = i
            break
    return levels[idx], str(levels[idx].get("name", f"level_{idx}")), progress


def spawn_pair(cfg, episode_id, training_steps=0):
    """Spawn two robots using the progressive curriculum."""
    arena = cfg["arena"]
    base_seed = int(cfg.get("experiment", {}).get("seed", 0))
    rng = np.random.default_rng(base_seed + 1009 * int(episode_id))
    z = float(arena.get("robot_spawn_z_m", 0.0980))

    level, level_name, progress = _curriculum_level(cfg, int(training_steps))
    if level is None:
        rmin = float(arena["spawn_radius_min_m"])
        rmax = float(arena["spawn_radius_max_m"])
        jitter = float(arena.get("spawn_angular_jitter_rad", 0.30))
        theta = rng.uniform(-math.pi, math.pi)
        theta2 = theta + math.pi + rng.uniform(-jitter, jitter)
        r0 = rng.uniform(rmin, rmax)
        r1 = rng.uniform(rmin, rmax)
        p0 = [float(r0 * math.cos(theta)), float(r0 * math.sin(theta)), z]
        p1 = [float(r1 * math.cos(theta2)), float(r1 * math.sin(theta2)), z]
        y0 = yaw_to_face(p0[:2], p1[:2]) + rng.uniform(-0.12, 0.12)
        y1 = yaw_to_face(p1[:2], p0[:2]) + rng.uniform(-0.12, 0.12)
        return p0, p1, y0, y1, level_name, progress

    dmin = float(level.get("distance_min_m", 0.35))
    dmax = float(level.get("distance_max_m", 0.55))
    midpoint_radius = float(level.get("midpoint_radius_m", 0.10))
    err0 = float(level.get("learner_heading_error_rad", 0.0))
    err1 = float(level.get("opponent_heading_error_rad", 0.0))

    separation = rng.uniform(dmin, dmax)
    axis = rng.uniform(-math.pi, math.pi)
    ux, uy = math.cos(axis), math.sin(axis)
    mr = midpoint_radius * math.sqrt(rng.random())
    mt = rng.uniform(-math.pi, math.pi)
    mx, my = mr * math.cos(mt), mr * math.sin(mt)
    p0 = [float(mx - 0.5 * separation * ux), float(my - 0.5 * separation * uy), z]
    p1 = [float(mx + 0.5 * separation * ux), float(my + 0.5 * separation * uy), z]
    y0 = yaw_to_face(p0[:2], p1[:2]) + rng.uniform(-err0, err0)
    y1 = yaw_to_face(p1[:2], p0[:2]) + rng.uniform(-err1, err1)
    return p0, p1, y0, y1, level_name, progress


def _contact_point_xyz(contact) -> Optional[Tuple[float, float, float]]:
    try:
        if hasattr(contact, "getPoint"):
            p = contact.getPoint()
        elif hasattr(contact, "point"):
            p = contact.point
        else:
            return None
        if p is None or len(p) < 3:
            return None
        return float(p[0]), float(p[1]), float(p[2])
    except Exception:
        return None


def _contact_xyz_list(node) -> List[Tuple[float, float, float]]:
    points: List[Tuple[float, float, float]] = []
    for cp in node.getContactPoints(True):
        xyz = _contact_point_xyz(cp)
        if xyz is not None:
            points.append(xyz)
    return points


def _robot_robot_contact(
    contacts0: Iterable[Tuple[float, float, float]],
    contacts1: Iterable[Tuple[float, float, float]],
    tolerance_m: float,
) -> bool:
    tol2 = float(tolerance_m) ** 2
    c0 = list(contacts0)
    c1 = list(contacts1)
    for a in c0:
        for b in c1:
            if ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) <= tol2:
                return True
    return False


def _safe_node_reset_physics(node) -> None:
    """Reset linear/angular velocity for a robot tree when Webots exposes the API."""
    try:
        node.resetPhysics()
    except Exception:
        # simulationResetPhysics below remains the portable fallback.
        pass


def main():
    cfg = load_config()
    arena = cfg["arena"]
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())
    receiver = supervisor.getDevice("receiver")
    receiver.enable(timestep)
    emitter = supervisor.getDevice("emitter")
    sumo0 = supervisor.getFromDef("SUMO_0")
    sumo1 = supervisor.getFromDef("SUMO_1")
    if sumo0 is None or sumo1 is None:
        raise RuntimeError("SUMO_0/SUMO_1 DEF nodes not found")

    radius = float(arena["ring_radius_m"])
    safety_floor_z = float(arena.get("safety_floor_top_z_m", 0.0))
    contact_z_tol = float(arena.get("safety_contact_z_tolerance_m", 0.012))
    radial_tol = float(arena.get("safety_contact_radial_tolerance_m", 0.005))
    fallback_center_z = float(arena.get("fallback_center_z_threshold_m", 0.064))
    contact_match_tol = float(arena.get("robot_contact_match_tolerance_m", 0.008))
    assist_window_steps = int(arena.get("contact_assist_window_steps", 200))
    max_steps = int(arena["max_episode_steps"])

    protocol = cfg.get("protocol", {})
    episode_end_retry_interval = max(1, int(protocol.get("episode_end_retry_interval_steps", 20)))
    reset_quiesce_steps = max(1, int(protocol.get("reset_quiesce_steps", 4)))
    reset_settle_steps = max(1, int(protocol.get("reset_settle_steps", 20)))
    reset_guard_steps = max(0, int(protocol.get("post_reset_integrity_guard_steps", 20)))
    reset_xy_tolerance = float(protocol.get("reset_position_tolerance_m", 0.025))
    reset_z_tolerance = float(protocol.get("reset_z_tolerance_m", 0.03))
    reset_reapply_limit = max(1, int(protocol.get("reset_max_reapply_attempts", 3)))

    required_done = set(int(x) for x in cfg.get("termination", {}).get("required_done_agents", [0]))
    raw_reset_ack = protocol.get("reset_required_ack_agents", "auto")
    if raw_reset_ack == "auto" or raw_reset_ack is None:
        reset_required_ack = set()
        for key, agent_cfg in (cfg.get("agents", {}) or {}).items():
            mode = str((agent_cfg or {}).get("mode", "")).lower()
            if mode in {"train", "eval", "evaluate", "evaluation"}:
                reset_required_ack.add(int(key))
        if not reset_required_ack:
            reset_required_ack = {0}
    else:
        reset_required_ack = set(int(x) for x in raw_reset_ack)
    reset_done_retry_interval = max(1, int(protocol.get("reset_done_retry_interval_steps", 20)))
    completed_agents = set()

    for node in (sumo0, sumo1):
        try:
            node.enableContactPointsTracking(timestep, True)
        except Exception:
            pass

    episode_id = 0
    episode_step = 0
    active = False
    contact_api_warning_printed = False
    last_robot_contact_step: Optional[int] = None
    robot_contact_steps = 0
    pending_episode_end: Optional[Dict[str, Any]] = None
    pending_episode_end_wait = 0
    current_curriculum_level = "uninitialized"
    current_curriculum_progress = 0.0
    last_reset_training_steps = 0
    reset_state: Optional[Dict[str, Any]] = None

    def outside_contact(node, contacts) -> Tuple[bool, Dict[str, Any]]:
        nonlocal contact_api_warning_printed
        try:
            for x, y, z in contacts:
                radial = math.hypot(x, y)
                on_lower_floor = z <= (safety_floor_z + contact_z_tol)
                outside_disk = radial >= (radius - radial_tol)
                if on_lower_floor and outside_disk:
                    return True, {
                        "criterion": "safety_floor_contact",
                        "contact_point": [x, y, z],
                        "contact_radius_m": radial,
                    }
        except Exception as exc:
            if not contact_api_warning_printed:
                print(f"[arena_supervisor] WARNING contact processing failed: {exc!r}; using center-height fallback")
                contact_api_warning_printed = True

        x, y, z = node.getPosition()
        if z <= fallback_center_z:
            return True, {
                "criterion": "center_height_fallback",
                "contact_point": None,
                "contact_radius_m": math.hypot(x, y),
            }
        return False, {"criterion": None, "contact_point": None, "contact_radius_m": None}

    def _apply_pose_and_zero(state: Dict[str, Any]) -> None:
        """Atomically re-pose both robots and clear all residual rigid-body motion."""
        p0, p1 = state["p0"], state["p1"]
        yaw0, yaw1 = state["yaw0"], state["yaw1"]

        # Clear pre-reset momentum before teleporting, then clear it again after.
        _safe_node_reset_physics(sumo0)
        _safe_node_reset_physics(sumo1)
        try:
            supervisor.simulationResetPhysics()
        except Exception:
            pass

        sumo0.getField("translation").setSFVec3f(p0)
        sumo1.getField("translation").setSFVec3f(p1)
        sumo0.getField("rotation").setSFRotation([0, 0, 1, yaw0])
        sumo1.getField("rotation").setSFRotation([0, 0, 1, yaw1])

        _safe_node_reset_physics(sumo0)
        _safe_node_reset_physics(sumo1)
        try:
            supervisor.simulationResetPhysics()
        except Exception:
            pass

    def _pose_ok(state: Dict[str, Any]) -> Tuple[bool, float, float]:
        p0a = list(sumo0.getPosition())
        p1a = list(sumo1.getPosition())
        p0e, p1e = state["p0"], state["p1"]
        xy0 = math.hypot(p0a[0] - p0e[0], p0a[1] - p0e[1])
        xy1 = math.hypot(p1a[0] - p1e[0], p1a[1] - p1e[1])
        zerr = max(abs(p0a[2] - p0e[2]), abs(p1a[2] - p1e[2]))
        centers_inside = math.hypot(p0a[0], p0a[1]) < radius and math.hypot(p1a[0], p1a[1]) < radius
        ok = max(xy0, xy1) <= reset_xy_tolerance and zerr <= reset_z_tolerance and centers_inside
        return bool(ok), max(xy0, xy1), zerr

    def begin_reset(target_episode: int, training_steps: int = 0) -> None:
        nonlocal active, reset_state, pending_episode_end, pending_episode_end_wait, last_reset_training_steps
        nonlocal current_curriculum_level, current_curriculum_progress

        target_episode = int(target_episode)
        training_steps = int(training_steps)
        last_reset_training_steps = training_steps
        p0, p1, yaw0, yaw1, curriculum_level, curriculum_progress = spawn_pair(
            cfg, target_episode, training_steps
        )
        current_curriculum_level = curriculum_level
        current_curriculum_progress = float(curriculum_progress)
        active = False
        pending_episode_end = None
        pending_episode_end_wait = 0
        reset_state = {
            "target_episode": target_episode,
            "training_steps": training_steps,
            "p0": p0,
            "p1": p1,
            "yaw0": yaw0,
            "yaw1": yaw1,
            "curriculum_level": curriculum_level,
            "curriculum_progress": float(curriculum_progress),
            "phase": "quiesce",
            "remaining": reset_quiesce_steps,
            "attempt": 1,
        }
        send_json(emitter, {
            "type": "reset_prepare",
            "episode_id": target_episode,
            "quiesce_steps": reset_quiesce_steps,
            "settle_steps": reset_settle_steps,
        })
        print(
            f"[reset_integrity] prepare episode={target_episode} "
            f"quiesce={reset_quiesce_steps} settle={reset_settle_steps}"
        )

    def advance_reset_state() -> None:
        nonlocal reset_state, episode_id, episode_step, active, last_robot_contact_step, robot_contact_steps
        nonlocal current_curriculum_level, current_curriculum_progress
        if reset_state is None:
            return

        phase = str(reset_state["phase"])
        if phase == "quiesce":
            reset_state["remaining"] = int(reset_state["remaining"]) - 1
            if int(reset_state["remaining"]) <= 0:
                _apply_pose_and_zero(reset_state)
                reset_state["phase"] = "settle"
                reset_state["remaining"] = reset_settle_steps
            return

        if phase == "settle":
            reset_state["remaining"] = int(reset_state["remaining"]) - 1
            if int(reset_state["remaining"]) > 0:
                return

            ok, xyerr, zerr = _pose_ok(reset_state)
            if not ok:
                attempt = int(reset_state.get("attempt", 1))
                if attempt < reset_reapply_limit:
                    reset_state["attempt"] = attempt + 1
                    print(
                        f"[reset_integrity] reapply episode={reset_state['target_episode']} "
                        f"attempt={attempt + 1}/{reset_reapply_limit} xy_error={xyerr:.4f} z_error={zerr:.4f}"
                    )
                    _apply_pose_and_zero(reset_state)
                    reset_state["remaining"] = reset_settle_steps
                    return
                raise RuntimeError(
                    f"Reset integrity failed for episode {reset_state['target_episode']}: "
                    f"xy_error={xyerr:.4f} m z_error={zerr:.4f} m after {attempt} attempts"
                )

            episode_id = int(reset_state["target_episode"])
            episode_step = 0
            last_robot_contact_step = None
            robot_contact_steps = 0
            current_curriculum_level = str(reset_state["curriculum_level"])
            current_curriculum_progress = float(reset_state["curriculum_progress"])
            p0a = list(sumo0.getPosition())
            p1a = list(sumo1.getPosition())
            # Two-phase release: do not let either controller move yet. The learner
            # must acknowledge reset_done, then the supervisor broadcasts reset_release.
            active = False
            reset_state["phase"] = "await_ack"
            reset_state["acked_agents"] = set()
            reset_state["wait_steps"] = 0
            reset_state["actual_p0"] = p0a
            reset_state["actual_p1"] = p1a
            payload = {
                "type": "reset_done",
                "episode_id": episode_id,
                "curriculum_level": current_curriculum_level,
                "curriculum_progress": current_curriculum_progress,
                "reset_integrity": "ok",
                "settle_steps": reset_settle_steps,
                "awaiting_ack": sorted(reset_required_ack),
            }
            reset_state["reset_done_payload"] = payload
            send_json(emitter, payload)
            print(
                f"[reset_integrity] posed episode={episode_id} "
                f"p0=({p0a[0]:+.3f},{p0a[1]:+.3f},{p0a[2]:.3f}) "
                f"p1=({p1a[0]:+.3f},{p1a[1]:+.3f},{p1a[2]:.3f}) "
                f"awaiting_ack={sorted(reset_required_ack)}"
            )
            return

        if phase == "await_ack":
            reset_state["wait_steps"] = int(reset_state.get("wait_steps", 0)) + 1
            if int(reset_state["wait_steps"]) % reset_done_retry_interval == 0:
                send_json(emitter, reset_state["reset_done_payload"])
            return

    print(
        "[arena_supervisor] ready "
        "ring_out=safety_floor_contact official_win=opponent_out_for_any_reason "
        f"causal_window={assist_window_steps * timestep / 1000.0:.2f}s "
        f"curriculum={cfg.get('curriculum', {}).get('profile', 'legacy')} "
        f"reset_protocol=quiesce_pose_settle_v1.12"
    )
    send_json(emitter, {"type": "supervisor_ready"})

    while supervisor.step(timestep) != -1:
        released_this_step = False
        for msg in read_messages(receiver):
            mtype = msg.get("type")
            if mtype == "reset_request":
                target = int(msg.get("target_episode", episode_id + 1))
                training_steps = int(msg.get("training_steps", last_reset_training_steps))
                if reset_state is not None:
                    if target == int(reset_state.get("target_episode", -1)):
                        # Retry from the agent while the phased reset is still settling
                        # or waiting for its acknowledgement.
                        if reset_state.get("phase") == "await_ack":
                            send_json(emitter, reset_state.get("reset_done_payload", {
                                "type": "reset_done", "episode_id": target
                            }))
                        else:
                            send_json(emitter, {
                                "type": "reset_prepare",
                                "episode_id": target,
                                "phase": reset_state.get("phase"),
                            })
                    continue
                if target > episode_id:
                    begin_reset(target, training_steps)
                elif target == episode_id:
                    # Recovery path if a controller missed the original broadcast.
                    send_json(emitter, {
                        "type": "reset_done",
                        "episode_id": episode_id,
                        "curriculum_level": current_curriculum_level,
                        "curriculum_progress": current_curriculum_progress,
                        "reset_integrity": "ok",
                        "settle_steps": reset_settle_steps,
                    })
                    if active:
                        send_json(emitter, {
                            "type": "reset_release",
                            "episode_id": episode_id,
                            "curriculum_level": current_curriculum_level,
                            "curriculum_progress": current_curriculum_progress,
                        })
            elif mtype == "reset_ack":
                ack_ep = int(msg.get("episode_id", -1))
                ack_agent = int(msg.get("agent_id", -1))
                if (reset_state is not None
                        and reset_state.get("phase") == "await_ack"
                        and ack_ep == int(reset_state.get("target_episode", -2))):
                    reset_state.setdefault("acked_agents", set()).add(ack_agent)
                    if reset_required_ack.issubset(set(reset_state.get("acked_agents", set()))):
                        p0a = list(reset_state.get("actual_p0", sumo0.getPosition()))
                        p1a = list(reset_state.get("actual_p1", sumo1.getPosition()))
                        active = True
                        released_this_step = True
                        send_json(emitter, {
                            "type": "reset_release",
                            "episode_id": episode_id,
                            "curriculum_level": current_curriculum_level,
                            "curriculum_progress": current_curriculum_progress,
                        })
                        print(
                            f"[arena_supervisor] reset episode={episode_id} "
                            f"p0=({p0a[0]:+.3f},{p0a[1]:+.3f},{p0a[2]:.3f}) "
                            f"p1=({p1a[0]:+.3f},{p1a[1]:+.3f},{p1a[2]:.3f}) "
                            f"curriculum={current_curriculum_level} progress={current_curriculum_progress:.3f} "
                            f"reset_integrity=ok released_after_ack={sorted(reset_state.get('acked_agents', set()))}"
                        )
                        reset_state = None
            elif mtype == "episode_end_ack":
                ack_ep = int(msg.get("episode_id", -1))
                if pending_episode_end is not None and ack_ep == int(pending_episode_end.get("episode_id", -2)):
                    pending_episode_end = None
                    pending_episode_end_wait = 0
            elif mtype == "agent_done":
                completed_agents.add(int(msg.get("agent_id", -1)))
                print(f"[arena_supervisor] done agents={sorted(completed_agents)} required={sorted(required_done)}")
                if required_done.issubset(completed_agents):
                    send_json(emitter, {"type": "shutdown"})
                    supervisor.simulationQuit(0)
            elif mtype == "agent_failed":
                print("[arena_supervisor] agent failure:", msg)
                supervisor.simulationQuit(2)

        if released_this_step:
            # Give all controllers one synchronized step to receive reset_release
            # before episode_step and contact/ring-out accounting begin.
            continue

        if reset_state is not None:
            advance_reset_state()
            continue

        if not active:
            if pending_episode_end is not None:
                pending_episode_end_wait += 1
                if pending_episode_end_wait % episode_end_retry_interval == 0:
                    send_json(emitter, pending_episode_end)
            continue

        episode_step += 1
        try:
            contacts0 = _contact_xyz_list(sumo0)
            contacts1 = _contact_xyz_list(sumo1)
        except Exception as exc:
            contacts0, contacts1 = [], []
            if not contact_api_warning_printed:
                print(f"[arena_supervisor] WARNING contact-point API unavailable: {exc!r}; fallback enabled")
                contact_api_warning_printed = True

        robot_contact = _robot_robot_contact(contacts0, contacts1, contact_match_tol)
        if robot_contact:
            last_robot_contact_step = episode_step
            robot_contact_steps += 1

        p0_now = list(sumo0.getPosition())
        p1_now = list(sumo1.getPosition())
        distance_m = math.hypot(p0_now[0] - p1_now[0], p0_now[1] - p1_now[1])
        send_json(emitter, {
            "type": "training_telemetry",
            "episode_id": episode_id,
            "robot_contact": bool(robot_contact),
            "distance_m": float(distance_m),
            "radii_m": {
                "0": float(math.hypot(p0_now[0], p0_now[1])),
                "1": float(math.hypot(p1_now[0], p1_now[1])),
            },
        })

        out0, detail0 = outside_contact(sumo0, contacts0)
        out1, detail1 = outside_contact(sumo1, contacts1)
        reason = None
        winner = None
        loser = None
        termination_criterion = None
        contact_point = None
        ring_out_cause = None
        valid_episode = True

        if (out0 or out1) and episode_step <= reset_guard_steps:
            # A robot cannot physically traverse from the configured spawn region to
            # the safety floor in this short interval. Treat this as reset-state
            # contamination, give no terminal reward, and force a clean retry.
            reason = "reset_integrity_failure"
            winner = None
            loser = None
            termination_criterion = "post_reset_guard_violation"
            ring_out_cause = "reset_artifact"
            valid_episode = False
            detail = detail0 if out0 else detail1
            contact_point = detail.get("contact_point")
            print(
                f"[reset_integrity] INVALID episode={episode_id} step={episode_step} "
                f"out0={out0} out1={out1}; discarded from publication evaluation"
            )
        elif out0 or out1:
            reason = "ring_out"
            if out0 and not out1:
                winner, loser = 1, 0
                termination_criterion = detail0.get("criterion")
                contact_point = detail0.get("contact_point")
            elif out1 and not out0:
                winner, loser = 0, 1
                termination_criterion = detail1.get("criterion")
                contact_point = detail1.get("contact_point")
            else:
                reason = "double_ring_out"
                winner, loser = None, None
                termination_criterion = "simultaneous_outside_contact_rematch"
                ring_out_cause = "simultaneous"
        elif episode_step >= max_steps:
            reason = "timeout"
            winner = None
            termination_criterion = "time_limit"

        if reason is not None:
            age_steps = None
            age_s = None
            if last_robot_contact_step is not None:
                age_steps = int(episode_step - last_robot_contact_step)
                age_s = float(age_steps * timestep / 1000.0)

            if reason == "ring_out":
                if age_steps is not None and age_steps <= assist_window_steps:
                    ring_out_cause = "contact_assisted"
                else:
                    ring_out_cause = "unforced_self_out"

            p0 = list(sumo0.getPosition())
            p1 = list(sumo1.getPosition())
            cp_text = ""
            if contact_point is not None:
                cp_text = f" contact=({contact_point[0]:+.3f},{contact_point[1]:+.3f},{contact_point[2]:+.3f})"
            age_text = "none" if age_s is None else f"{age_s:.3f}s"
            print(
                f"[arena_supervisor] end episode={episode_id} reason={reason} "
                f"winner={winner} loser={loser} criterion={termination_criterion} cause={ring_out_cause} "
                f"valid={int(valid_episode)} last_robot_contact_age={age_text}{cp_text} "
                f"p0=({p0[0]:+.3f},{p0[1]:+.3f},{p0[2]:.3f}) "
                f"p1=({p1[0]:+.3f},{p1[1]:+.3f},{p1[2]:.3f})"
            )
            episode_end_payload = {
                "type": "episode_end",
                "episode_id": episode_id,
                "winner": winner,
                "loser": loser,
                "reason": reason,
                "valid_episode": bool(valid_episode),
                "termination_criterion": termination_criterion,
                "ring_out_cause": ring_out_cause,
                "last_robot_contact_age_steps": age_steps,
                "last_robot_contact_age_s": age_s,
                "robot_contact_steps": robot_contact_steps,
                "ring_out_contact_point": contact_point,
                "steps": episode_step,
                "positions": {"0": p0, "1": p1},
                "out_details": {"0": detail0, "1": detail1},
                "reset_guard_steps": reset_guard_steps,
            }
            send_json(emitter, episode_end_payload)
            pending_episode_end = episode_end_payload
            pending_episode_end_wait = 0
            active = False


if __name__ == "__main__":
    main()
