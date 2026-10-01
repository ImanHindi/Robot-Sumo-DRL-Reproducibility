# v1.7 Physical Hardware Build Plan (after RL environment freeze)

The v1.7 Webots model is the mechanical reference that should be converted to CAD before fabrication. Do not build the old 120 mm box model.

## High-level architecture

- 195 x 195 mm low-profile chassis.
- 3.0 kg maximum final assembled mass including battery and accessories.
- Smooth 195 mm front wedge, approximately 75 mm long.
- Two 60 mm high-grip drive wheels.
- Four opponent range directions: front, rear, left, right.
- Four downward reflectance/edge sensors: front-left, rear-left, front-right, rear-right.
- A microcontroller handles motor PWM, edge sampling and emergency stop; Raspberry Pi (or equivalent) can run the frozen RL policy if desired.
- A visible status LED should indicate ready/running/stopped state.

## Important deployment interface

The trained policy expects six normalized values:

`[front, rear_flag, left, right, left_edge_flag, right_edge_flag]`

The four physical edge sensors are aggregated in the microcontroller:

- `left_edge_flag = front_left_edge OR rear_left_edge`
- `right_edge_flag = front_right_edge OR rear_right_edge`

This preserves the Webots policy interface while increasing real edge-detection coverage.

## Sensor choice

Do not lock the final sensor brand until the target competition rules and ring material are confirmed. The Webots model is sensor-type agnostic at the policy level. For real hardware, select range and reflectance sensors with sufficient update rate, field of view and sunlight/IR immunity, then calibrate their transfer functions against the real Dohyo.

## Wedge material and safety

The Webots wedge is a smooth stainless-steel reference. A real blade must not be sharpened or capable of scratching/gouging the Dohyo. Use a polished/rounded leading edge and verify it during inspection against the target tournament's current rules.

## Final sim-to-real loop

1. Generate CAD from the v1.7 geometry.
2. Choose motors, battery, wheel compound, motor drivers and sensors.
3. Measure the built robot's actual mass, wheel radius, wheelbase, center of mass and top speed.
4. Update Webots with measured values.
5. Re-run a short calibration/fine-tuning phase if physical parameters differ materially.
6. Freeze the final policy before competition evaluation.
