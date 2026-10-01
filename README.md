# Competitive Learning and Adaptation in Autonomous Robot-Sumo

Reproducibility repository for the study **“Competitive Learning and Adaptation in Autonomous Robot-Sumo: Multi-Seed Deep Reinforcement Learning Benchmarking and Asymmetric Veteran–Novice Self-Play.”**

This release contains the frozen Webots Robot-Sumo environment, reinforcement-learning controllers, benchmark and self-play experiment code, analysis scripts, analysis-ready data, final manuscript figure generator, and the associated manuscript snapshot.

## Scope

The study uses an official-geometry 3 kg Robot-Sumo simulation in Webots and evaluates four deep reinforcement-learning algorithms under a controlled two-stage training protocol:

- A2C
- PPO
- DQN
- RecurrentPPO

Each algorithm is trained with five independent seeds. The final benchmark contains 20 trained policies and 6,000 deterministic evaluation bouts across static, scripted, and aggressive opponents. A second experiment studies asymmetric PPO self-play between a pretrained veteran and a newly initialized novice across five paired seeds and five Stage-3 checkpoints.

## Frozen experimental versions

- **Core benchmark environment:** v1.12 — clean reset and translational-attack accounting
- **Asymmetric self-play extension:** v1.14 — veteran–novice paired self-play
- **Benchmark run:** `20260913_220238_benchmark_v112`
- **Self-play run:** `20260918_074640_asymmetric_selfplay_v114`

The v1.14 source preserves the validated v1.12 physical environment, robot design, observations, actions, reward formulation, reset integrity, and external-opponent evaluation protocol.

## Repository structure

```text
controllers/          Webots robot and arena-supervisor controllers
worlds/               Webots world
protos/               Robot PROTO definition
config/               Frozen experiment configuration
experiments/          Stage 1, Stage 2, and Stage 3 experiment YAML files
scripts/              Training, evaluation, benchmark, and self-play orchestration
analysis/             Benchmark and self-play analysis
tools/               Final manuscript figure generator
data/benchmark/      Analysis-ready benchmark data
data/selfplay/       Analysis-ready asymmetric self-play data
 docs/                Methodological and validation documentation
 deploy/raspberry_pi/ Sim-to-real deployment reference
 manuscript/          Manuscript and highlights snapshot
 metadata/            Machine-readable release and archive metadata
```

## Environment

The validated Python environment is defined in both `environment.yml` and `requirements.txt`. The experiment was run with Python 3.9 and Stable-Baselines3 2.6.0 / sb3-contrib 2.6.0. Webots is required for simulation execution.

## Reproducing the main benchmark

```bash
python scripts/benchmark_algorithms.py \
  --algorithms A2C PPO DQN RecurrentPPO \
  --seeds 0 1 2 3 4 \
  --stage1-steps 200000 \
  --stage2-steps 300000 \
  --eval-episodes 100 \
  --name benchmark_v112
```

Central analysis:

```bash
python analysis/analyze_benchmark.py --root <benchmark_run_directory>
```

## Reproducing asymmetric veteran–novice self-play

The self-play experiment uses the corresponding PPO Stage-2 veteran model from each benchmark seed.

```bash
python scripts/asymmetric_selfplay_master.py \
  --benchmark-root <benchmark_run_directory> \
  --seeds 0 1 2 3 4 \
  --stage3-steps 500000 \
  --checkpoint-every 50000 \
  --checkpoint-eval-steps 100000 200000 300000 400000 500000 \
  --intermediate-episodes 50 \
  --final-h2h-episodes 100 \
  --external-episodes 100 \
  --name asymmetric_selfplay_v114
```

Central analysis:

```bash
python analysis/analyze_asymmetric_selfplay.py \
  --root <selfplay_run_directory> \
  --benchmark-root <benchmark_run_directory>
```

## Reproducing manuscript figures

The compact repository contains all analysis-ready data required for the benchmark and self-play summary figures. Figure 3 additionally uses raw Stage-2 training episode logs from the complete benchmark archive distributed with the archival reproducibility deposit.

```bash
python tools/generate_manuscript_figures.py \
  --benchmark-root <benchmark_run_directory> \
  --selfplay-results <selfplay_run_directory>/paper_results \
  --output figures
```

For analysis-ready data only:

```bash
python tools/generate_manuscript_figures.py \
  --benchmark-paper-results data/benchmark \
  --selfplay-results data/selfplay \
  --output figures \
  --skip-figure3
```

## Primary result files

Benchmark:

- `data/benchmark/all_evaluation_episodes.csv` — all 6,000 evaluation bouts
- `data/benchmark/per_seed_summary.csv` — seed-level summaries
- `data/benchmark/algorithm_summary_mean_sd_ci95.csv` — algorithm-condition means, SDs, and 95% CIs
- `data/benchmark/statistical_tests_win_rate.csv` — seed-level nonparametric tests

Self-play:

- `data/selfplay/selfplay_head_to_head_per_seed_checkpoint.csv`
- `data/selfplay/selfplay_head_to_head_summary_ci95.csv`
- `data/selfplay/selfplay_external_generalization_per_seed.csv`
- `data/selfplay/selfplay_training_trajectory.csv`
- `data/selfplay/selfplay_statistical_tests.csv`

## Reproducibility principles

- The trained seed is the independent unit for between-algorithm inference.
- Official win/loss outcome is separated from contact-assisted causal attribution.
- Invalid post-reset artifacts are excluded by the validated reset-integrity protocol.
- Policy observations remain hardware-reproducible; privileged Supervisor telemetry is used only for training-time shaping and analysis where documented.
- Benchmark and self-play versions are frozen and identified in `metadata/experiment_manifest.json`.

## License

Source code is released under the MIT License. See `LICENSE`.

## Author

**Iman Hindi**  
Department of Industrial Engineering, School of Engineering Technology  
Al Hussein Technical University, Amman, Jordan
