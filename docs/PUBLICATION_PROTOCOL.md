# v1.12 publication-integrity additions

For the final benchmark, use only valid episodes released through the verified reset protocol. Any `reset_integrity_failure` is a diagnostic artifact, receives zero terminal reward, is stored separately, and must not enter win/loss/draw statistics. Report translational attack using `translational_forward_ratio`, with `straight_forward_ratio` and `curved_forward_ratio` available to explain policy style. Do not combine v1.11 and v1.12 runs in inferential statistics.

---

# Publication Protocol for a Journal-Quality Study

## Core experimental questions

A defensible manuscript can be organized around three distinct questions:

1. Does staged curriculum training (stationary opponent -> moving opponent) improve learning efficiency and final fighting performance?
2. Which RL algorithm provides the best performance and robustness under the same observation/action/reward design?
3. Under asymmetric self-play, why can a newly initialized opponent become more aggressive or outperform a previously trained veteran policy?

These questions should be analyzed separately rather than mixing all training histories into one comparison.

## Minimum recommended repetitions

Use at least five independent seeds per algorithm; ten is stronger if compute permits. Keep the same total environment-step budget for every algorithm. Report wall-clock training time separately because on-policy, off-policy, and recurrent methods have different computational cost.


## Arena fidelity and two evaluation layers

All v1.6 benchmark runs use the standard 3 kg-class geometry encoded in `worlds/sumo_research.wbt`: outer diameter 1.54 m, height 0.05 m, 0.05 m inward white Tawara, 20 cm x 2 cm brown Shikiri-sen at ±10 cm, and a 3.60 m square surrounding safety area. Record these values explicitly in the methods section.

Keep two evaluation layers separate:

1. **RL benchmark evaluation** — fixed 3000-step (15 s at 5 ms) episodes and deterministic/randomized test seeds. This is useful for equal-compute algorithm comparison.
2. **Tournament-style evaluation** — a separate final experiment reflecting the official battle procedure more closely (three matches within three minutes, Yuko-based outcome, official starting procedure). Do not call the 15 s benchmark episodes official tournament matches.

The current ring-out detector uses Webots Supervisor contact points and declares a ring-out when a robot part contacts the lower safety area outside the Dohyo footprint. A conservative center-height fallback is retained only as a fail-safe. State the simulation criterion explicitly in the paper.

For sim-to-real claims, replace the generic 12 cm x 12 cm chassis dimensions with measured dimensions, wheel radius, mass distribution, motor limits, sensor poses, and friction/calibration data from the physical robot. The current model is 3.0 kg and lies within the 20 cm x 20 cm maximum footprint, but it is not automatically a geometric twin of the physical robot.

## Fixed evaluation protocol

After training, freeze the model. Evaluate it for at least 100 matches per fixed opponent condition. Use identical deterministic spawn generation for algorithms being compared. Include at least:

- stationary opponent
- standard scripted moving opponent
- aggressive scripted moving opponent

For learned-vs-learned head-to-head comparisons, evaluate both robot-slot assignments to reduce systematic side/controller bias.

## Model selection

Do not choose the final paper model by looking at the final test tournament. If hyperparameter tuning is added later, use separate validation conditions/seeds and reserve the reported test tournament for final evaluation only.

## Statistics

Report per-seed means, standard deviations, and 95% confidence intervals. The included analysis uses a Kruskal-Wallis omnibus test and Holm-corrected pairwise Mann-Whitney tests on per-seed win rates. For a stronger paper, consider also reporting effect sizes and bootstrap confidence intervals.

## Figures recommended for the manuscript

- training win rate versus environment steps
- evaluation win rate with 95% CI for each algorithm
- episode duration / time-to-win distribution
- aggression-index comparison
- engagement and edge-risk comparison
- head-to-head veteran vs novice results
- optional tournament matrix if all trained algorithms fight one another

## Ablations that would strengthen the paper

If compute permits, add controlled ablations:

- no Stage 1 curriculum vs Stage 1 + Stage 2 curriculum
- sparse terminal reward vs revised shaped reward
- feed-forward PPO vs RecurrentPPO
- normalized sensor observations vs legacy raw observations
- fixed opponent vs opponent-pool/self-play training

## Reproducibility reporting

Record and publish:

- Webots version
- Python, Torch, Stable-Baselines3 and sb3-contrib versions
- random seeds
- world timestep and episode limit
- observation normalization
- action mapping
- reward equation
- algorithm hyperparameters
- training step budgets
- evaluation opponent definitions
- number of matches
- hardware specifications if training time is compared

The enhanced package writes a runtime manifest and raw episode CSVs so these details can be reconstructed later.


## Ring-out event definition (v1.6)

For publication runs, ring-out is defined by physical contact with the lower safety area outside the Dohyo, using Webots Supervisor contact points including descendant wheel solids. This is deliberately different from the development-only v1.5 center-radius threshold. Log the terminating contact coordinates and criterion for auditability.

---

## v1.7 mechanical/reward freeze addendum

All publication runs should use the v1.7 professional identical robot and should start from new randomly initialized policies. Do not mix models produced by the earlier 120 mm box robot with v1.7 because mechanical dynamics and wheel radius changed.

Official outcome and causal attribution must be reported separately. A win is an official win whenever the opponent leaves the Dohyo for any reason. Additionally report the fraction of wins classified as `contact_assisted` versus `unforced_self_out`. This prevents a high win rate caused primarily by opponent self-outs from being interpreted as superior pushing performance.

The main algorithm-comparison table should include at minimum: win rate, forced/contact-assisted win rate, unforced-win rate, loss rate, draw rate, median episode duration, aggression index, engagement ratio and edge-risk ratio. Keep the same reward coefficients, robot model, surface condition and opponent suite for every compared algorithm and seed.

The v1.7 light Dohyo scuffs are part of the frozen benchmark environment. If surface-condition randomization is studied later, treat it as a separate robustness experiment using an identical surface-seed schedule across algorithms.

## v1.9 curriculum amendment

The v1.8 500k PPO sanity test converged to a near-all-draw policy with negligible engagement. v1.9 therefore freezes the physical simulator but revises the learning curriculum. Stage 1 progresses from close frontal attack to orientation/approach to randomized search within a single training process; Stage 2 progresses from nearby moving-opponent interaction to randomized moving search. All compared algorithms receive the same 200k + 300k environment-step budget and the same curriculum boundaries. Evaluation uses an independent randomized start profile. Report curriculum configuration, sensor abstraction, reward weights, and per-seed behavioral metrics alongside win rate.


## v1.10 benchmark-opponent safety amendment

The v1.9 quick evaluation exposed an evaluation confound: scripted/aggressive opponents could occasionally self-exit without recent RL-agent contact, inflating nominal win rate. v1.10 leaves the learned-agent task, curriculum, physical model, and terminal rules unchanged but replaces the scripted opponent's non-directional edge recovery with a deterministic four-sensor directional guard. Front-edge detections command an inward reverse escape; rear-edge detections command an inward forward escape; lateral detections select a turn away from the boundary; ambiguous states rotate in place; a short post-edge guard suppresses immediate re-attack.

For publication runs, report both overall win rate and win causality (`forced_win_rate`, `unforced_win_rate`). A high unforced-opponent self-out fraction in scripted/aggressive evaluation should be treated as a benchmark-integrity warning rather than evidence of policy strength.


## v1.11 attack-learning and action-audit amendment

The v1.10 full-budget PPO sanity run demonstrated a learning local optimum: opponent detection remained high while forward attack/contact were almost absent and evaluation was dominated by draws. v1.11 therefore keeps the complete physical simulator and benchmark-opponent safety logic frozen but revises credit assignment.

The policy observation remains six-dimensional and hardware-reproducible. During training only, Supervisor telemetry supplies true center-to-center distance, physical robot contact, and radial positions for reward shaping. These variables are **not** policy inputs and must be described as privileged training information in the Methods section.

For publication, log and report action frequencies, dominant action, empirical action entropy, attack ratio, physical-contact ratio, forward-contact pressure ratio, first-contact events, outward-push progress, forced/contact-assisted win rate, and draw rate. The v1.11 action map is fixed at 18 discrete actions and is automatically audited before the supplied benchmark launchers.

Do not combine v1.10 and v1.11 trained-policy performance in the main algorithm-comparison statistics because the learning reward/curriculum differs. The physical-model hashes may be used to demonstrate that the world and robot PROTO were unchanged.
