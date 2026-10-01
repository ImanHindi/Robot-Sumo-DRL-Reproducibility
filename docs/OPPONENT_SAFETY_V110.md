# v1.10 Scripted-Opponent Edge Safety

## Why this patch exists

The v1.9 quick test showed nominal wins against scripted/aggressive opponents that were mostly classified as `unforced_self_out`. Those wins are valid under official match scoring, but they are a poor benchmark signal when the opponent simply drives itself out with no recent contact from the learned agent.

The root cause was in the scripted controller, not the Webots arena or ring-out detector. The controller combined all four ground sensors into a single `edge` flag and always commanded reverse. That is correct when the **front** sensors reach the boundary, but it is the wrong escape direction when the **rear** sensors reach it.

## v1.10 directional guard

- Front edge detected -> reverse immediately into the Dohyo.
- Rear edge detected -> drive forward immediately into the Dohyo.
- Left-side edge -> subsequent turn is to the right.
- Right-side edge -> subsequent turn is to the left.
- Ambiguous front+rear edge -> stop translating and rotate in place.
- Edge response starts in the same controller cycle as detection.
- After the boundary clears, the robot continues the inward escape briefly, turns away from the boundary, then uses a short rotation-only guard before normal attack/search resumes.
- Search remains pure in-place rotation so an unseen opponent does not cause random translational drift toward the edge.

The standard and aggressive profiles retain different attack/search speeds. The aggressive profile therefore remains a harder opponent, but both profiles use the same safety principle.

## What is deliberately unchanged

The official 3 kg Dohyo geometry, used-surface appearance, robot dimensions and mass, wedge, wheel geometry, contact physics, range/edge sensors, RL observation space, reward function, v1.9 curriculum, ring-out criterion, causal attribution, and reset/episode-end handshakes are unchanged.

## Acceptance check

Run `RUN_QUICK_TEST_VISIBLE.bat`. The console should show:

```text
[scripted] ... edge_guard=directional_v1.10 ...
```

Then inspect evaluation causality. A contact-assisted ring-out is legitimate. Repeated `cause=unforced_self_out` for the scripted/aggressive opponent indicates the guard still needs calibration and the full benchmark should not be started.

After the quick test passes, run `RUN_PPO_SANITY.bat` (200k Stage 1 + 300k Stage 2, seed 0, 50 episodes per evaluation opponent). Only then proceed to the five-seed benchmark.
