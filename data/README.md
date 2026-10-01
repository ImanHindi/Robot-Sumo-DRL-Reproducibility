# Data

## Benchmark

`benchmark/all_evaluation_episodes.csv` contains 6,000 deterministic evaluation bouts from four algorithms, five seeds, and three opponent conditions.

`benchmark/per_seed_summary.csv` is the primary seed-level analysis table.

`benchmark/algorithm_summary_mean_sd_ci95.csv` contains aggregated descriptive statistics.

`benchmark/statistical_tests_win_rate.csv` contains Kruskal-Wallis and Holm-corrected pairwise Mann-Whitney results.

## Self-play

`selfplay/selfplay_head_to_head_per_seed_checkpoint.csv` contains role-swapped veteran-novice checkpoint outcomes.

`selfplay/selfplay_head_to_head_summary_ci95.csv` contains checkpoint means and confidence intervals.

`selfplay/selfplay_external_generalization_per_seed.csv` contains pre/post veteran and post-self-play novice external-opponent results.

`selfplay/selfplay_training_trajectory.csv` contains Stage-3 training trajectories.

`selfplay/selfplay_statistical_tests.csv` contains paired inferential analyses.
