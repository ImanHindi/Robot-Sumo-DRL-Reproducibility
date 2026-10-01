# Results summary

## Multi-algorithm benchmark

Across five independent seeds, PPO produced the highest descriptive mean win rates against the scripted and aggressive opponents and the largest number of contact-assisted forced wins. However, Kruskal-Wallis tests on seed-level win rates were not significant for static (p = 0.418), scripted (p = 0.956), or aggressive (p = 0.454) conditions. The study therefore does not claim statistically significant algorithm superiority.

The benchmark revealed substantial between-seed variability and qualitatively different learned strategies. PPO showed the strongest descriptive conversion of physical engagement into successful ring-outs. A2C exhibited comparatively high forward activity, DQN showed greater exploration together with increased edge risk and loss rate, and RecurrentPPO tended toward draw-dominated conservative behavior.

## Asymmetric veteran-novice self-play

At 500k Stage-3 steps, the paired novice-minus-veteran official win-rate difference was not significant (paired Wilcoxon p = 0.655). The novice temporarily closed or reversed the gap at intermediate checkpoints in some seeds, but a consistent novice takeover did not emerge.

External generalization showed lower post-self-play veteran forced-win rates than pre-self-play benchmark performance. The mean post-minus-pre changes were -0.020 against static, -0.088 against scripted, and -0.096 against aggressive opponents. With five pairs, Wilcoxon tests remained underpowered; the aggressive-condition bootstrap interval for the mean difference was below zero in the archived analysis.

These results support an interpretation of seed-sensitive competitive strategy discovery and opponent-specific co-adaptation rather than monotonic improvement from additional self-play.
