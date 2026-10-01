# v1.12 — Clean Reset + Translational Attack Metrics

v1.12 is a focused integrity/measurement revision built on v1.11. The Webots world, robot PROTO, arena geometry, sensors, wheel geometry, opponent-safety controller, curriculum, and reward magnitudes are preserved.

## 1. Phased reset protocol

The reset is no longer a single teleport followed immediately by `reset_done`.

1. `reset_prepare` freezes the learner and benchmark opponent.
2. A short quiescence window lets old motor commands reach zero.
3. The supervisor clears residual rigid-body velocity, applies both robot poses, and clears physics again.
4. A settling window is simulated with both controllers held at zero velocity.
5. The supervisor verifies the actual robot positions against the requested poses.
6. `reset_done` is sent only after the pose passes verification.
7. The learning agent returns `reset_ack`.
8. The supervisor broadcasts `reset_release`, after which both controllers may act.

This prevents one controller from beginning the next bout while the other is still waiting for a reset packet.

The default protocol is configured in `config/base_experiment.yaml`:

- quiescence: 4 Webots steps
- settle: 20 Webots steps
- XY pose tolerance: 0.025 m
- Z tolerance: 0.03 m
- maximum pose reapplication attempts: 3
- reset acknowledgements: automatically required from every RL-controlled `train`/`eval` agent
- post-reset integrity guard: 20 episode steps

If a robot is detected on the safety floor during the post-reset guard, the episode is labeled `reset_integrity_failure`, receives zero terminal reward, and is excluded from publication evaluation. The event is saved separately in `evaluation_invalid_reset_artifacts.csv`.

## 2. Translational attack semantics

v1.11 correctly audited the 18 discrete actions, but its behavior metrics treated only actions 0–2 (`forward_*`) as forward motion. Actions 6–11 (`left_*`, `right_*`) also command both wheels forward and therefore produce curved forward translation.

v1.12 classifies wheel commands physically:

- `straight_forward`: both wheels positive and action label is `forward`
- `curved_forward`: both wheels positive and action label is `left` or `right`
- `translational_forward`: straight OR curved forward
- spin: wheel signs oppose each other

Attack, sustained contact pressure, outward-push progress, and safe forward search use `translational_forward`. Reward magnitudes are unchanged from v1.11.

The CSVs now report all of:

- `forward_ratio` — backward-compatible alias for translational-forward motion in v1.12
- `translational_forward_ratio`
- `straight_forward_ratio`
- `curved_forward_ratio`
- `attack_ratio`
- `contact_forward_ratio`
- `push_outward_progress_ratio`

Exact per-action ratios remain available, so the paper can still distinguish the learned steering preference from true straight-line charging.

## 3. Publication rule

Do not combine v1.11 and v1.12 runs in the final benchmark. Reset semantics and attack-credit accounting changed. Start fresh training from Stage 1 for the v1.12 publication dataset.
