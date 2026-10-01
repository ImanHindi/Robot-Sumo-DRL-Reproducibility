# Package validation — v1.12

Validation performed before packaging:

- all Python files parse and compile successfully;
- all YAML files parse successfully;
- `package_version` is `v1.12` in both base and active configuration;
- Webots world SHA-256 remains unchanged from v1.11: `ca5ab769de17be797a6df8cfb91a6bd1864d056602bc56b71edced047689ce28`;
- `SumoRobot2.proto` SHA-256 remains unchanged from v1.11: `39f3704e9deba1b36a3939bc773c6696701f7599c44abc95362ee28e9887203d`;
- the reset supervisor contains the complete `reset_prepare -> quiesce -> pose/zero physics -> settle -> pose verification -> reset_done -> reset_ack -> reset_release` protocol;
- reset acknowledgement defaults to `auto`, requiring every RL-controlled `train`/`eval` agent to acknowledge the verified pose before release;
- scripted and frozen controllers contain reset-hold logic and remain at zero wheel velocity until `reset_release`;
- immediate post-reset safety-floor events are labeled `reset_integrity_failure`, receive zero terminal reward, and are excluded from valid evaluation counts;
- invalid evaluation attempts are stored separately in `evaluation_invalid_reset_artifacts.csv` and automatically retried;
- wheel-sign unit checks confirm actions 0–2 are straight forward, actions 6–11 are curved forward with both wheels positive, and actions 12–17 are spins;
- reward unit checks confirm straight and curved forward commands receive identical attack/contact-forward credit under the same physical state, while spins/backward commands do not;
- invalid reset terminal events return `(0, 0, 0)` terminal/shaping reward;
- synthetic benchmark-analysis validation confirms invalid episodes are filtered and the new `translational_forward_ratio`, `straight_forward_ratio`, and `curved_forward_ratio` propagate into paper summaries and figures;
- v1.11/v1.10 curriculum, reward magnitudes, opponent edge safety, policy observation, arena, and robot physical model remain otherwise unchanged.

Webots itself is not installed in the packaging environment. Therefore `RUN_QUICK_TEST_VISIBLE.bat` is still required as the simulator-side validation. In that run, verify that resets print `reset_integrity=ok` and `released_after_ack`, and that no repeated short carryover episodes appear.
