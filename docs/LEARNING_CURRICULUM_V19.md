# v1.9 Learning Curriculum

v1.9 keeps the v1.8 official 3 kg Dohyo, professional identical robot geometry, drivetrain, contact physics, ring-out criterion, causal win attribution, reset protocol, and reliable episode-end handshake. It changes only the learning formulation that produced the 500k-step PPO "safe draw" local optimum.

## Stage 1 — stationary opponent, progressive difficulty (200k steps total)

The curriculum advances automatically inside one Webots/SB3 training process. This is important for DQN because the replay buffer is not discarded between 1A/1B/1C.

- **1A Attack close (0–30%)**: static opponent 0.30–0.38 m away, almost directly in front. Goal: learn detect → drive forward → engage → push.
- **1B Orient and approach (30–65%)**: distance 0.36–0.52 m and learner heading error up to ±0.75 rad. Goal: learn turn toward target before attacking.
- **1C Search static (65–100%)**: distance 0.45–0.70 m and full learner heading randomization. Goal: learn scanning/search without sacrificing edge safety.

The supervisor prints the active curriculum level at every reset.

## Stage 2 — moving scripted opponent (300k steps total)

- **2A Moving near (0–40%)**: moderate distance and orientation error.
- **2B Moving search (40–100%)**: larger separation and full heading randomization.

The same Stage-1 model continues into Stage 2. For DQN, v1.9 also saves/restores `replay_buffer.pkl` across the Stage-1 → Stage-2 transition.

## Evaluation

Evaluation uses a separate randomized start profile (`evaluation_randomized`) so results are not restricted to the easy training geometry.

## Sensor abstraction

The physical robot geometry is unchanged. Opponent detection is made more realistic/useful for search by using multi-ray range sensing:

- Front: 0.85 m, 9 rays, 0.42 rad aperture
- Left/right: 0.70 m, 7 rays, 0.52 rad aperture
- Rear: 0.65 m, 5 rays, 0.50 rad aperture

The policy observation remains six-dimensional for deployment compatibility:

`[front proximity, rear detected, left proximity, right proximity, left edge, right edge]`

## Reward hierarchy

Terminal match outcome remains dominant:

- official win: +12
- official loss: -12
- timeout/draw: -3
- contact-assisted ring-out bonus: +3
- unforced self-out extra penalty: -2

Dense shaping is deliberately smaller. It rewards safe search, scanning, increasing target proximity, correct turning, frontal engagement and forward attack. It penalizes edge danger, backing away without an edge threat, and moving away from a detected target.

This is designed to eliminate the v1.8 local optimum where doing little and accepting a timeout was safer than fighting.

## Required validation before the full benchmark

Run `RUN_PPO_SANITY.bat` first. Approve the full benchmark only if the 500k PPO model shows clear non-zero target detection, engagement/attack, robot contact, and at least some decisive wins/losses rather than approximately 100% timeout draws.
