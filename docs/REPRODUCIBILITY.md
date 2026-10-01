# Reproducibility record

## Experimental units

### Benchmark

- Algorithms: A2C, PPO, DQN, RecurrentPPO
- Independent seeds: 0, 1, 2, 3, 4
- Stage 1: 200,000 environment steps against a stationary opponent under the progressive curriculum
- Stage 2: 300,000 environment steps against the scripted moving opponent
- Total training budget: 500,000 environment steps per policy
- Deterministic evaluation: 100 bouts against each of static, scripted, and aggressive opponents
- Total trained policies: 20
- Total evaluation bouts: 6,000

### Asymmetric self-play

- Algorithm: PPO
- Paired seeds: 0, 1, 2, 3, 4
- Veteran initialization: corresponding PPO Stage-2 model from the benchmark
- Novice initialization: fresh PPO policy
- Stage 3 simultaneous self-play: 500,000 steps
- Head-to-head checkpoints: 100k, 200k, 300k, 400k, 500k
- Role-swapped deterministic evaluation
- Final external generalization: static, scripted, and aggressive opponents

## Frozen simulator formulation

The frozen v1.12 configuration is stored in `config/base_experiment.yaml` and records the official-geometry Dohyo, professional 3 kg robot reference, sensor layout, action mapping, reward coefficients, reset-integrity logic, causal ring-out attribution, opponent profiles, and curriculum.

The policy observation is six-dimensional and hardware-reproducible. The action space contains 18 differential-drive commands. Curved forward motion is explicitly included in translational-attack accounting.

## Statistical unit

A trained seed is the independent unit for inferential tests. Individual evaluation bouts are repeated measurements within a trained policy and are not treated as independent algorithm-level samples.

## Data layers

`data/benchmark/` and `data/selfplay/` contain analysis-ready publication data. The complete raw benchmark and self-play run directories, including model/checkpoint artifacts and raw training logs, are preserved in the archival deposit listed in `metadata/raw_archive_manifest.csv`.

## Figure provenance

`tools/generate_manuscript_figures.py` generates the manuscript figures from the publication datasets. Figure 3 requires the raw Stage-2 training logs in the complete benchmark run. All other data-driven figures can be regenerated directly from the compact repository data.
