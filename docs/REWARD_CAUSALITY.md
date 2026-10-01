# Official Outcome vs. RL Credit Assignment — v1.7

## If red drives out by itself, does blue still win?

Yes for the official match outcome. Under the current All Japan Robot-Sumo rules, a contestant receives a Yuko point when the opponent touches any part outside the Dohyo **for any reason**. Therefore, if red leaves the Dohyo without blue touching it, red is the loser and blue is the official winner.

That official rule is preserved exactly in v1.7.

## Why we also record causal attribution

For reinforcement learning, an official win does not necessarily mean that the winner caused the ring-out. If blue is far away and red self-outs, giving blue the same *behavioral credit* as a successful push can make interpretation of aggression and strategy difficult.

v1.7 therefore separates two concepts:

1. **Official outcome** — winner/loser used for win rate and the main terminal reward.
2. **Causal attribution** — whether there was recent physical robot-to-robot contact before the losing robot touched the safety area.

A ring-out is logged as:

- `contact_assisted` if robot-to-robot contact occurred within the configured recent-contact window (default 1.0 s), or
- `unforced_self_out` if there was no such recent contact.

The classification never changes the official winner.

## Reward design

The v1.7 reward scale is deliberately rebalanced so dense shaping cannot dominate the match result.

- official win: +10
- official loss: -10
- timeout: -1
- contact-assisted win bonus: +2
- unforced opponent self-out bonus: +0 beyond the normal +10 official win
- own unforced self-out penalty: -1 beyond the normal -10 loss
- small per-step search/engagement/edge shaping terms remain much smaller than terminal outcome

This means a learning blue agent still learns that surviving an opponent self-out is a valid win, but it receives extra credit only for a contact-assisted ring-out. A static/scripted non-learning opponent does not optimize or consume an RL reward at all.

## Publication metrics

Each episode now records:

- `winner`, `loser`, `win`, `loss`, `draw`
- `ring_out_cause`
- `forced_win`, `unforced_win`
- `last_robot_contact_age_steps`, `last_robot_contact_age_s`
- `robot_contact_steps`
- `official_terminal_reward`
- `causal_shaping_reward`
- exact ring-out contact coordinates

The benchmark analysis additionally produces forced-vs-unforced win-rate summaries and figures. This allows the paper to distinguish a policy that wins by actively forcing opponents out from a policy that benefits from opponent self-outs.
