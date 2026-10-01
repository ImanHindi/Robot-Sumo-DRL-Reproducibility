# Arena specification used in v1.6

## Modeled class

The simulation now uses the **3 kg autonomous robot-sumo geometry** used by the All Japan
Robot-Sumo Tournament family of rules.

### Dohyo geometry represented in Webots

| Item | v1.6 value |
|---|---:|
| Outer dohyo diameter | 1.54 m |
| Outer dohyo radius | 0.77 m |
| Dohyo height | 0.05 m |
| White border (Tawara) width | 0.05 m, drawn inward from the circumference |
| Black interior radius | 0.72 m |
| Starting lines | two brown lines, 0.20 m long x 0.02 m wide |
| Safety area (Yochi) | 3.60 m x 3.60 m |

The white border is **inside** the 1.54 m outer diameter and is part of the valid dohyo surface.
The v1.4 arena was not competition-accurate: it used a 1.00 m black disk inside a 1.50 m white
disk, creating a 0.25 m visible white annulus, and the physical platform was only 0.005 m high.
Those values should not be used for a paper claiming a standard 3 kg robot-sumo arena.

## Robot model represented

The current Webots robot has a total modeled mass of 3.0 kg and a 0.12 m x 0.12 m chassis.
That is within the 3 kg-class maximum 0.20 m x 0.20 m starting footprint, but it is intentionally
not enlarged to the maximum. For a sim-to-real paper, replace the 0.12 m dimensions with the
measured dimensions of the physical robot if they differ.

## What is still a research protocol rather than a full tournament reproduction

The arena **geometry** is now competition-based, but the default RL experiment does not claim to
implement every tournament procedure. In particular:

- training uses seeded randomized mirrored starting poses rather than referee placement at the
  shikiri lines;
- the default RL episode horizon is 3000 x 5 ms = 15 s, not the complete 3-minute match format;
- the study records one simulated bout/episode at a time rather than implementing the complete
  two-Yuko match scoring procedure and referee judgments such as Yusei;
- collision friction and the exact physical robot geometry must ultimately be calibrated against
  the real robot/ring before claiming quantitative sim-to-real fidelity.

For the journal study, this is acceptable if described as a controlled RL benchmark on an
official-geometry dohyo. If the paper claims direct reproduction of official competition matches,
a separate official-match evaluation protocol should be added.


## Ring-out criterion

v1.6 does **not** declare a robot out merely because its center crosses radius 0.77 m. The white Tawara is inside the Dohyo and a Yuko point is awarded when a robot touches the outside safety area. The supervisor therefore checks contact points of the chassis and wheel descendants; a lower-floor contact outside the circular footprint ends the episode.

## v1.7 used-surface rendering

v1.7 keeps the same official geometry but changes appearance from a pristine synthetic disk to a maintained, lightly used competition surface. Dark-gray cosmetic scuffs are rendered slightly above the black top and are excluded from the collision bounding object, so the physical surface remains flat. Because Webots infrared sensors are reflectance-sensitive, these marks provide mild sensor variation while remaining far darker than the white Tawara. Severe gouges or exposed substrate are not modeled.
