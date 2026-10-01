# v1.14 Asymmetric Veteran-versus-Novice Self-Play Protocol

## Scientific purpose

This is an **add-on experiment** to the frozen v1.12 benchmark. It does not replace or retune the 4-algorithm benchmark. It tests a separate question: whether a freshly initialized learner can adapt rapidly when trained against an already competent veteran that continues to learn at the same time.

The design is motivated by the observed phenomenon that the initially inexperienced robot may later develop more aggressive or effective strategies than the pretrained robot. The experiment is designed to determine whether that effect is reproducible across independent seeds, whether it reflects genuine contact-assisted wins rather than opponent self-outs, and whether it generalizes beyond the co-adapting opponent.

## Primary research question

**RQ-SP1.** During asymmetric self-play, does a novice PPO agent close or reverse the competitive-performance gap relative to a PPO veteran that enters self-play with a 500,000-step training head start and continues learning concurrently?

## Secondary research questions

**RQ-SP2.** Do novice and veteran policies exhibit different trajectories in aggression, attack commitment, physical contact, forward pressure, and edge risk?

**RQ-SP3.** If the novice reaches or exceeds the veteran head-to-head, does the improvement transfer to frozen external opponents (static, standard scripted, aggressive scripted), or is it opponent-specific co-adaptation?

**RQ-SP4.** Does continued self-play improve or degrade the veteran's original external performance, indicating adaptation versus catastrophic forgetting?

## Hypotheses

- **H1:** The initial veteran advantage decreases with Stage-3 self-play experience.
- **H2:** At least some novice seeds reach head-to-head parity or superiority before 500,000 Stage-3 steps.
- **H3:** Novice adaptation is accompanied by increases in attack/contact/forced-win behavior rather than only unforced opponent self-outs.
- **H4:** External evaluation differentiates generalizable improvement from opponent-specific exploitation.

These are predeclared hypotheses. The experiment must be reported even if the novice does not surpass the veteran.

## Agents and initialization

For each of five paired replicates (seeds 0–4):

- **Veteran (Agent 0):** the corresponding PPO `stage2_scripted/agent_0/model_final.zip` from the completed v1.12 benchmark. Each veteran has already received 200k Stage-1 + 300k Stage-2 steps = 500k pretraining steps.
- **Novice (Agent 1):** a newly initialized PPO policy with the same architecture and hyperparameters and no pretrained weights.
- Both agents use the same six-dimensional onboard-sensor observation, 18-action discrete motor command set, reward structure, 3 kg robot geometry, arena, ring-out attribution, and clean reset protocol as frozen v1.12.
- During Stage 3 both agents learn simultaneously. The veteran is **not frozen**; this is deliberately asymmetric initialization with symmetric continued adaptation.

## Stage-3 training budget

Each pair trains for **500,000 additional simultaneous environment steps**. This gives the novice a Stage-3 exposure budget equal in length to the veteran's entire original 500k pretraining budget, while the veteran retains its 500k-step head start and continues to adapt.

Stage-3 spawn conditions use the frozen `evaluation_randomized` distribution rather than an easier curriculum. Checkpoints are stored every 50k steps. Deterministic evaluations are performed at 100k, 200k, 300k, 400k, and 500k Stage-3 steps.

## Head-to-head evaluation

At each evaluation checkpoint:

- Both policies are frozen for deterministic evaluation.
- Each matchup is evaluated in **both robot slots** using the same spawn seed.
- Intermediate checkpoints: 50 episodes per slot (100 fights per pair/checkpoint).
- Final 500k checkpoint: 100 episodes per slot (200 fights per pair).
- Primary outcome: role-swapped head-to-head win rate.
- Causal outcome: role-swapped contact-assisted/forced-win rate.
- Draw rate and unforced self-outs remain explicit.

Role swapping prevents a robot-slot or spawn-position artifact from being mistaken for adaptation.

## External generalization evaluation

After Stage 3, veteran and novice final policies are independently evaluated for 100 episodes against each frozen external opponent:

1. static,
2. standard scripted,
3. aggressive scripted.

The evaluation seed is exactly the seed used by the original v1.12 benchmark (`100000 + training_seed`), enabling a paired veteran-pre versus veteran-post comparison on the same deterministic spawn sequence. The original v1.12 veteran evaluation is read directly from the completed benchmark and is never overwritten.

## Primary and secondary metrics

Primary:
- official win rate,
- contact-assisted forced-win rate,
- novice-minus-veteran head-to-head win-rate difference.

Secondary:
- draw and loss rates,
- aggression index,
- translational-forward ratio,
- attack ratio,
- contact ratio,
- contact-forward ratio,
- outward-push progress ratio,
- edge-risk ratio.

A novice is not interpreted as "stronger" merely because it receives higher shaped reward. Competitive outcomes and causal ring-out attribution remain primary.

## Independent unit and statistics

The **trained seed/pair** is the independent experimental unit (`n = 5`), not the hundreds of fights generated by each trained policy. Episode counts characterize each frozen policy precisely but do not artificially inflate inferential sample size.

At the final checkpoint the analysis reports:
- paired novice and veteran win rates for all five seeds,
- paired mean and median difference,
- seed-bootstrap 95% confidence interval,
- exact/automatic paired Wilcoxon signed-rank test,
- per-seed parity-crossing checkpoint.

With only five independent pairs, statistical power is limited and effect sizes/trajectory consistency are emphasized over dichotomous p-value claims.

## Interpretation safeguards

1. **No cherry-picking:** all five PPO veteran seeds from the original benchmark are included, including weak and strong veterans.
2. **No post-result retuning:** reward, sensors, action set, physics, opponent controllers, and PPO hyperparameters remain frozen.
3. **Causality:** forced wins are separated from unforced opponent self-outs.
4. **Generalization:** a novice that only beats its co-adapting veteran is not automatically considered globally stronger.
5. **Forgetting:** veteran post-self-play external performance is compared with veteran pre-self-play performance.
6. **Role bias:** every direct matchup is evaluated in both agent slots.
7. **Reset integrity:** v1.12 clean-reset validation remains active; invalid reset artifacts are excluded and retried automatically.

## Expected paper contribution

The experiment allows the manuscript to move beyond a static algorithm benchmark toward a study of **competitive adaptation under asymmetric prior experience**. Whether the novice surpasses the veteran or fails to do so is scientifically informative: both outcomes quantify how prior training, exploration, co-adaptation, and strategy convergence interact in embodied adversarial RL.
