# Professional 3 kg Autonomous Sumo Robot Reference — v1.7

## Why the robot is being frozen before the final RL study

The mechanical design must be fixed before publication training. Wheel diameter, wheelbase, mass distribution, wedge geometry, sensor locations, ground clearance and friction all change the state transition dynamics that the RL agent learns. Therefore the final paper should **not** train a policy on a generic box robot and then redesign the physical robot afterward without retraining.

v1.7 defines a hardware-buildable reference robot and uses the **same geometry for both competitors**. Red and blue are only identification colors.

## Competition envelope

- Reference class: autonomous 3 kg Robot-Sumo.
- Inspection footprint: **195 mm x 195 mm**. This deliberately leaves 5 mm total tolerance under a nominal 200 mm x 200 mm inspection frame.
- Simulated total mass: **3.00 kg**.
- Height: low profile; no competition-height assumption is required.

## Mechanical architecture

### Front wedge / kas7a

- Width: **195 mm**.
- Length: **75 mm**.
- Simulated thickness: **2.5 mm**.
- Material reference: smooth stainless steel.
- Inclination: approximately 13.8 degrees.
- Leading edge: near the Dohyo surface but **not sharpened**.

For a real build, the front lip should be polished/rounded so it can get under an opponent without cutting, gouging or scratching the Dohyo. The current All Japan rules prohibit components that damage the Dohyo and prohibit blade materials that can break apart in operation/contact. Final fabrication must be checked against the exact event inspection rules.

### Chassis and armor

- Main chassis shell: 190 mm wide x 125 mm long x 48 mm high.
- Side armor rails remain inside the 195 mm inspection envelope.
- Rear bumper remains inside the same envelope.
- Rear low-friction skid: PTFE-like reference pad for stable three-point support.
- Reference center of mass is low and slightly rearward: `[0, +28 mm, -10 mm]` in the robot local frame.

The model intentionally keeps the center of mass close to the driven wheels so traction is not wasted by loading the wedge excessively.

### Drivetrain

- Two identical drive wheels.
- Wheel diameter: **60 mm**.
- Wheel width: **28 mm**.
- Nominal motor limit: **22 rad/s**.
- Nominal motor maximum torque in Webots: **1.5 N.m per wheel**.
- RL speed levels: 7 / 12 / 18 rad/s, equivalent to approximately 0.21 / 0.36 / 0.54 m/s ideal wheel-rim speed.

The tire model represents a high-grip compliant competition tire, not an adhesive device.

## Sensors

The policy interface remains six-dimensional for deployment simplicity, but the physical robot now has better safety coverage.

Opponent sensing:

- front range sensor
- rear range/obstacle sensor
- left side range sensor
- right side range sensor

Edge sensing:

- front-left downward IR sensor
- rear-left downward IR sensor
- front-right downward IR sensor
- rear-right downward IR sensor

The two left edge sensors are OR-combined into one policy flag; the two right edge sensors are OR-combined into one policy flag. The observation is therefore still:

`[front, rear_flag, left, right, left_edge_flag, right_edge_flag]`

This arrangement is more realistic for a fast 3 kg sumo robot because it can detect a boundary when reversing as well as when charging forward.

## Identical opponents

`SUMO_0` and `SUMO_1` instantiate the exact same `SumoRobot2.proto`. They share dimensions, mass, wheel geometry, motor parameters, sensor locations, collision geometry and center of mass. Only the identification color, controller ID and runtime policy differ.

This symmetry is important for fair head-to-head and algorithm-comparison experiments.

## Physical build later

The Webots design is intended to become the mechanical reference for the physical robot. Before fabrication, convert the geometry into CAD and perform an engineering pass for motor mounting, battery retention, electronics, structural stiffness, fasteners, service access and inspection tolerance. The physical robot should then be weighed and measured; those real values should be pushed back into Webots for the final sim-to-real validation model.
