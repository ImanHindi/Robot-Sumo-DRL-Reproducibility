# Physics and geometry validation notes — v1.6

The earlier v1.3/v1.4 packages successfully removed the repeated Webots physics-step warnings,
but a later visual inspection exposed two modeling issues that had to be corrected before the
simulation could be treated as publication-grade.

## Important correction to the earlier v1.3 note

The earlier note incorrectly stated that a Webots `Cylinder` is aligned with local Y. In Webots
R2025a the `Cylinder` central axis is local **Z**. Rotating a Z-axis cylinder around Z therefore
does not reorient its axle. v1.5 corrects this explicitly.

## v1.5 drivetrain corrections

1. **Wheel collision/visual cylinder orientation**
   - Native cylinder axle: local Z.
   - Required wheel axle for this robot: local X.
   - Each wheel Solid is now rotated `0 1 0 1.5708`, i.e. +90° about local Y.

2. **Motor sign and physical forward direction**
   - The robot's modeled front is local -Y (all front sensors point toward -Y).
   - Both hinge axes are now local +X.
   - Positive motor velocity therefore rolls both wheels toward local -Y, so the controller action
     called `forward` is physically forward.

3. **Left/right handedness**
   - With forward = -Y and up = +Z, the robot's left side is local +X and right side is local -X.
   - The left wheel and left side sensor are now on +X.
   - The right wheel and right side sensor are now on -X.
   - Ground sensors already used this handedness and are unchanged.

4. **Mass and contact model retained**
   - 2.8 kg chassis + two 0.1 kg wheels = 3.0 kg total.
   - Chassis collision size remains 0.12 x 0.12 x 0.08 m.
   - Motor limit remains 22 rad/s; action speeds remain 8/14/20 rad/s.
   - Zero-bounce softened contact properties from v1.3 are retained.

5. **Visibility-only graphics**
   - A thin colored top cap and white front marker were added to each robot.
   - They have no Physics or boundingObject nodes, so they do not affect mass, collision, or RL dynamics.

## Arena correction

v1.5 also replaces the simplified training disk with an official-geometry 3 kg dohyo:
1.54 m outer diameter, 5 cm height, and a 5 cm inward white border. See
`ARENA_SPECIFICATION.md` for the exact geometry and which aspects of the tournament rules are
and are not represented.

## Required validation before long runs

Run `RUN_QUICK_TEST_VISIBLE.bat` first. Confirm visually that:

- both red and blue robots are clearly visible;
- positive `forward` behavior moves toward each robot's white front marker;
- wheels are vertical and roll on the dohyo rather than lying flat;
- the white border is narrow (5 cm), not a large white annulus;
- robots reset on the black interior and fall onto the lower gray safety area after leaving the dohyo;
- no repeated physics-step warnings occur.

Discard models trained with v1.4 or earlier for scientific comparison because v1.5 changes the
drivetrain kinematics and arena geometry.


## v1.6 event-detection correction

The v1.5 geometry/kinematics remain unchanged. Only terminal event detection is revised: the supervisor uses robot contact points on the lower safety floor rather than center radius alone. This affects when terminal rewards are issued, so v1.6 starts a new scientifically consistent result series.
