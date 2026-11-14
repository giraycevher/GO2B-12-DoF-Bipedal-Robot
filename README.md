# GO2B — 12-DoF Bipedal Robot

**ZMP Preview Control · DCM Push Recovery · LQR Stabilization · Rough-Terrain Walking · MuJoCo**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://github.com/giraycevher/GO2B-12-DoF-Bipedal-Robot/actions/workflows/python-package.yml/badge.svg)
![Simulator](https://img.shields.io/badge/simulator-MuJoCo-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Paper](https://img.shields.io/badge/paper-UMTIK%202026-lightgrey)

![GO2B robot](assets/v2_robot.png)

**GO2B** is a custom-built, 12-degree-of-freedom (12-DoF) bipedal robot designed for **low-budget, resource-constrained hardware** such as the Teensy 4.1 microcontroller. This repository contains its simulation and control software, written in Python on top of the [MuJoCo](https://mujoco.org/) physics engine.

The central design idea is simple:

> **Do the heavy math offline. Keep the real-time loop tiny.**

Walking trajectories are computed ahead of time (ZMP preview control). A lightweight online loop (LQR stabilizer + DCM push recovery) only has to correct *errors*, which is cheap enough for a microcontroller.

> **Status:** this repository is a **simulation-only** release. The microcontroller firmware is not included, and sim-to-real transfer on the physical prototype is future work. See [Limitations](#limitations) and [Roadmap](#roadmap).

---

## Table of Contents

1. [Academic publication](#academic-publication)
2. [Key features](#key-features)
3. [How it works (in plain language)](#how-it-works-in-plain-language)
4. [System architecture](#system-architecture)
5. [Repository structure](#repository-structure)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Simulation results](#simulation-results)
9. [Rough-terrain walking](#rough-terrain-walking)
10. [Hardware platform](#hardware-platform)
11. [Limitations](#limitations)
12. [Roadmap](#roadmap)
13. [Troubleshooting](#troubleshooting)
14. [Contributing](#contributing)
15. [Citation](#citation)
16. [References](#references)
17. [License](#license)

---

## Academic Publication

The methods in this repository are based on our conference paper:

> Ünver, H. Ö., Lotfi, B., Eser, G. C., Ecevit, O. (2026). **"Development of Model Predictive Controlled 12-DoF Bipedal Robot: A Two-Phase Control Architecture Tailored for Low-Budget Bipedal Systems."**
> *21st International Conference on Machine Design and Production (UMTIK 2026), Istanbul, Türkiye.*
> [Read the paper (PDF)](assets/Development_of_Model_Predictive_Controlled_12-DoF_Bipedal_Robot.pdf)


## My Contributions

I was responsible for the main software and control-development activities of
the GO2B project. My work included:

- Building the Python and MuJoCo simulation environment
- Developing the ZMP preview-control pipeline
- Developing the online LQR stabilization algorithm
- Implementing DCM-based corrective foot-placement logic
- Implementing analytical inverse kinematics for the 6-DoF legs
- Developing the embedded software for the physical robot
- Integrating and calibrating the IMU and foot-contact sensing systems
- Developing telemetry, data-logging, and experiment tools
- Running and analyzing the simulation experiments
- Preparing the technical figures and results
- Writing the complete conference paper and preparing the final manuscript
- Refactoring, documenting, and maintaining the public codebase

The conference publication was produced with the technical, academic, and
supervisory contributions of all listed authors. The manuscript itself,
including its mathematical exposition, methodology, figures, and results
discussion, was written by Giray Cevher Eser.

---

## Key Features

- **Two-phase hierarchical control**
  - **Offline:** ZMP (Zero-Moment Point) preview control, built on the Linear Inverted Pendulum Model (LIPM) and a discrete algebraic Riccati equation (DARE), with a receding horizon. It generates stable walking trajectories.
  - **Online:** a fast LQR-based stabilizer corrects balance in real time.
- **DCM (Capture Point) push recovery.** The robot withstands external pushes (a 30 N lateral push in our tests). It computes the divergent component of motion (DCM) error and adjusts the foot landing position to keep its balance. An optional torso-tilt-to-ankle feedback is also available.
- **Analytical inverse kinematics.** A geometric, matrix-free solver for 6-DoF legs instead of a Jacobian-based solver. An optional damped-least-squares (DLS) refinement is available with `--refine-ik`.
- **Modular design.** Controllers, planner and kinematics are separate packages. By design, only `sim/mujoco_interface.py` (plus the viewer and logging glue in `main.py`) touches MuJoCo, so the control code can later be ported to firmware.
- **Built-in experiment tooling.** Push tests with visualization (arrow or real ball impact), video recording, telemetry plots, CSV traces and a JSON metadata file for every run.
- **Rough-terrain walking.** `main_terrain.py` walks the robot over procedurally generated, forest-floor-like ground (soil undulations, roots, stones, pits) with terrain-aware swing trajectories, slope-adapted foot orientation, early-touchdown detection and an ankle balance loop. See [Rough-terrain walking](#rough-terrain-walking).
- **Tested.** 44 automated tests (IK/FK, preview control, terrain, kinematic dry run, short simulations) run on every push via GitHub Actions.

---

## How It Works (in Plain Language)

You do not need a robotics PhD to follow this project. Here are the five ideas it is built on.

### 1. Walking is controlled falling

A biped is never truly standing still while it walks. It is like balancing a broomstick on your palm: to keep the stick up, you keep moving your hand underneath it. Robots do the same with their feet.

### 2. LIPM — the "broomstick" model

Instead of simulating every link and motor, we simplify the robot to **one point mass (the center of mass, CoM) on a massless stick at constant height** $z_c$. The motion of this model is:

$$\ddot{x} = \omega^2 \,(x - p), \qquad \omega = \sqrt{g / z_c}$$

Here $x$ is the CoM position, $p$ is where the robot "pushes" on the ground (the ZMP, below), and $g$ is gravity. It is a crude model, but it is simple enough to solve exactly, which is why it is so useful.

### 3. ZMP and preview control — "headlights on the road"

The **Zero-Moment Point (ZMP)** is the point on the ground where the robot's tipping moment is zero. Think of it as the center of pressure under your feet. For stable walking it must stay **inside the support area (the foot)**.

**Preview control** is like driving at night: you steer for the curve your headlights show *ahead*, not the one under your wheels. The controller knows where the ZMP *will* need to be over the next fraction of a second and plans the CoM motion to match. Mathematically, this is a Riccati-equation-based optimal controller (the DARE in `preview_controller.py`). It is solved offline, because it is the heavy part.

### 4. LQR — a pre-computed autopilot

Even a perfect plan meets reality: friction, modeling errors, small bumps. The **LQR (Linear Quadratic Regulator)** is a feedback law whose gains are computed once by solving an optimization problem. Online, the robot only applies those fixed gains to its measured error. That is cheap and fast, which is exactly what a low-cost microcontroller needs.

### 5. DCM / Capture Point — "where should I step to not fall?"

The **Divergent Component of Motion (DCM)**, also known as the capture point, is defined as:

$$\xi = x + \dot{x} / \omega$$

It answers one question: *"If I planted my foot right here, would I stop falling?"* Under the LIPM, the DCM is exactly that spot. If someone shoves the robot, the DCM jumps away from the foot. The controller measures this error and **moves the next landing position toward it**, so the robot recovers by taking a corrective step rather than fighting the push with its ankles.

### Bonus: analytical inverse kinematics

Given a desired foot position and orientation, **inverse kinematics (IK)** computes the six joint angles of the leg. Our solver uses closed-form geometry. It is fast, deterministic and needs no matrix inversions, so it is easy to port to firmware. A Jacobian-based refinement (`--refine-ik`) is optional.

---

## System Architecture

```mermaid
flowchart LR
    subgraph OFF["Phase 1 - Offline (heavy compute)"]
        P[Step planner<br/>footsteps + swing legs] --> Z[ZMP preview control<br/>LIPM + DARE]
        Z --> R[CoM / ZMP reference<br/>trajectory]
    end

    subgraph ON["Phase 2 - Online (real-time loop)"]
        R --> IK[Analytical IK<br/>CoM + feet to 12 joint angles]
        IK --> S[(MuJoCo simulation<br/>or real robot)]
        S -- "torso tilt, CoM state, contacts" --> L[LQR stabilizer]
        S -- "CoM position and velocity" --> D[DCM push recovery]
        L --> IK
        D -- "foot landing correction" --> IK
    end

    P -. swing-leg path .-> IK
```

| Phase | Where it runs | What it does |
| --- | --- | --- |
| **1. Offline** | Laptop / PC | Plans footsteps, solves the preview-control problem, produces the reference CoM/ZMP trajectory. |
| **2. Online** | Real-time loop (target: Teensy 4.1 at 500 Hz, i.e. a 2 ms period) | Tracks the reference, corrects balance errors (LQR), reacts to pushes (DCM), converts targets to joint angles (IK). |

---

## Repository Structure

```
├── main.py                            # Flat-ground walking: main loop and command-line interface
├── main_terrain.py                    # Rough-terrain walking: main loop and command-line interface
├── robot_walk.xml                     # MuJoCo robot model (full CAD meshes, 2.99 kg)
├── requirements.txt                   # Python dependencies
├── pytest.ini                         # Test configuration
├── config/parameters.py               # All constants (gains, limits, terrain, simulation parameters)
├── kinematics/leg_ik.py               # Analytical inverse and forward kinematics
├── controllers/preview_controller.py  # LIPM and DARE preview controller
├── controllers/push_recovery.py       # DCM capture-point logic and stabilization
├── planners/step_planner.py           # Flat-ground footsteps and swing legs
├── planners/terrain_planner.py        # Terrain-aware footsteps, swing legs and ankle balance
├── sim/mujoco_interface.py            # MuJoCo wrapper (robot, sensors, video recorder, push visualization)
├── sim/terrain.py                     # Forest-floor heightfield generator
├── utils/data_logger.py               # Telemetry, logging and plotting
├── tools/build_model.py               # Utility to fix CAD-exported XML files
├── tools/run_terrain_tests.py         # Batch of terrain experiments -> results table
├── tests/                             # Automated tests (pytest)
├── docs/                              # Methods note and code guide (PDF)
└── assets/                            # STL meshes for the model, images and media
```

Every script finds `robot_walk.xml` and its packages on its own, so you can run it from any folder (`python path/to/main.py`). All paths are resolved against the repository root (`config.parameters.PROJECT_ROOT`).

**Suggested reading order for newcomers:**

1. `config/parameters.py` — see every tunable number in one place.
2. `kinematics/leg_ik.py` — how joint angles are computed from foot poses.
3. `planners/step_planner.py` — how footsteps and swing legs are generated.
4. `controllers/preview_controller.py` — the LIPM + DARE trajectory generator.
5. `controllers/push_recovery.py` — the DCM logic.
6. `main.py` — how everything is stitched into one loop.
7. `planners/terrain_planner.py` and `main_terrain.py` — the rough-terrain extension.

---

## Installation

Python 3.10 or newer is required (current MuJoCo releases do not support 3.9).

```bash
git clone https://github.com/giraycevher/GO2B-12-DoF-Bipedal-Robot.git
cd GO2B-12-DoF-Bipedal-Robot
pip install -r requirements.txt
```

`requirements.txt` installs `numpy`, `scipy`, `mujoco`, `matplotlib`, and `imageio` with `imageio-ffmpeg` (the last two are only needed for video recording).

**Quick sanity check** (no window, no plot, 10 simulated seconds):

```bash
python main.py --headless --time 10 --no-plot
python -m pytest            # run the automated tests (pip install pytest)
```

---

## Usage

```bash
python main.py                       # visual simulation with the free camera
python main.py --headless            # no window; prints telemetry and saves a plot
```

### Push tests

```bash
python main.py --push 30 --push-axis lat       # 30 N lateral push (default visualization: red arrow)
python main.py --push 30 --push-viz ball       # a real ball impact instead of an arrow
python main.py --push 15 --push-axis fwd       # forward push
python main.py --push 30 --no-dcm              # disable DCM push recovery for comparison
python main.py --push 0                        # no push (unperturbed walking)
```

By default the push is applied at t = 6 s for 0.1 s.

### Recording a video

```bash
python main.py --record walk.mp4
python main.py --record walk.mp4 --headless --width 1920 --height 1080 --fps 60
```

### Main options

| Option | Description |
| --- | --- |
| `--headless` | Run without a viewer window |
| `--xml PATH` | MuJoCo model to load (the default is set in `config/parameters.py`) |
| `--zhip` | Hip-center height above ground in meters |
| `--step-len` | Step length |
| `--push`, `--push-axis`, `--push-viz` | Push force in N, direction (`lat` or `fwd`), and visualization (`arrow`, `ball`, `none`) |
| `--no-dcm`, `--k-dcm` | Disable the DCM feedback, or set its gain (0 = off, 1.0 = full) |
| `--ankle-fb`, `--ankle-fb-sign` | Enable torso-tilt to ankle feedback; use `-1` if it makes balance worse |
| `--refine-ik` | Refine the analytical IK solution with damped least squares |
| `--time` | Maximum simulation time in seconds |
| `--speed` | Playback speed (1.0 = real time, 0 = as fast as possible) |
| `--record FILE.mp4`, `--fps`, `--width`, `--height` | Video recording settings |
| `--no-plot` | Do not generate the telemetry plot |

Run `python main.py --help` for the full list. Gains, limits and simulation constants live in `config/parameters.py`.

### Outputs

- A one-line status report per simulated second in the terminal (CoM progress, hip height, swing leg, contacts, DCM error and correction).
- A telemetry plot (unless `--no-plot` is set).
- `main_walk_trace_push<F>n.csv`: raw trace with the time, 12 joint commands and actual joint angles, base pose, CoM position and velocity, and foot contacts.
- `main_walk_trace_push<F>n.metadata.json`: run settings, exit reason, software versions, and a walked/fell result.

### Reproducing the two headline experiments

```bash
python main.py --push 0                          # Scenario 1: undisturbed walking
python main.py --push 30 --push-axis lat         # Scenario 2: 30 N lateral push, DCM on
python main.py --push 30 --push-axis lat --no-dcm  # same push, DCM off (comparison)
```

---

## Simulation Results

These are qualitative demonstrations of the two core behaviors. A systematic evaluation (success rates over randomized trials, push-recovery envelope, ablations, latency and noise robustness) is planned.

Reproduced with the current code and model (`python main.py --headless`): undisturbed walking covers 0.72 m in 14 s; with a 30 N, 0.1 s lateral push the robot keeps walking (0.66 m, DCM on) and falls with `--no-dcm`.

**Scenario 1: walking without disturbance.** The robot tracked the offline-generated CoM trajectory on flat ground. Although the LIPM assumption neglects leg masses, the simulation showed stable heel-strike and swing-leg transients.

![Scenario 1](https://github.com/user-attachments/assets/616be702-e658-4a58-8e95-aec3bd1930fb)

![Scenario 1 graph](assets/result1.png)

**Scenario 2: 30 N lateral push.** A lateral force of 30 N was applied to the torso for 0.1 s during the walking cycle. The controller detected the drift. Instead of over-torquing the ankles, it moved the landing position of the swing leg toward the direction of the fall (capture point strategy).

![Scenario 2](https://github.com/user-attachments/assets/777ee1dc-1aa6-4a46-839b-686c971681ad)

![Scenario 2 graph](assets/result2.png)

---

## Rough-Terrain Walking

`main_terrain.py` uses the same ZMP preview controller and analytical IK as `main.py` and adds a terrain-adaptation layer (`planners/terrain_planner.py`) so that the robot walks over **uneven, forest-floor-like ground** generated by `sim/terrain.py`.

![Rough-terrain walk](assets/rough_terrain_walk.gif)

Full video: [`assets/rough_terrain_walk.mp4`](assets/rough_terrain_walk.mp4) (24 steps at `--rough 1.0`).

![Rough-terrain result](assets/rough_terrain_result.png)

*Left: terrain height map with the measured footholds and CoM path. Top right: commanded sole height of each foot; the flat segments show the ground height each foot ended up standing on. Bottom right: measured torso roll and pitch.*

### Running it

```bash
python main_terrain.py                        # rough = 0.6, viewer window
python main_terrain.py --rough 1.0            # harder terrain
python main_terrain.py --rough 1.0 --seed 7   # a different random terrain
python main_terrain.py --headless             # no window; summary + plot (walk_terrain.png)
python main_terrain.py --rough 1.0 --record walk.mp4
python tools/run_terrain_tests.py             # batch of experiments -> results table
python planners/terrain_planner.py            # kinematic dry run, no physics
```

| Option | Description |
| --- | --- |
| `--rough R` | Roughness: 0 = flat, 0.6 = default, 1.0 = hard, 1.5 = very hard |
| `--seed N` | A different random terrain |
| `--blind` | Do not use the height map; adapt only through contact sensing |
| `--no-slope` | Keep the soles level instead of matching the local slope |
| `--no-touchdown` | Disable early-touchdown detection |
| `--no-ankle` | Disable the ankle balance loop |
| `--dcm` | Enable DCM foot placement (off by default, see below) |
| `--step-time`, `--step-len`, `--step-height`, `--steps` | Gait timing and geometry |
| `--push F`, `--push-axis` | Push of F newtons for 0.1 s at t = 6 s |
| `--dump FILE.npz` | 100 Hz log of every controller signal |

At startup the script ray-casts the MuJoCo heightfield and checks it against the controller's height map; it stops if they differ by more than 2 mm. The terrain scene is written to `robot_terrain.xml` next to `robot_walk.xml` (ignored by git).

### What "rough" means at this scale

The leg is only 17.5 cm long, so the terrain is scaled to the robot. `sim/terrain.py` builds a MuJoCo heightfield from four layers: broad soil undulations, fine gravel/leaf texture, roots crossing the path as thin ridges, and small stones and pits. At `--rough 1.0` the height range is about −14 / +18 mm and 99 % of the slopes are below 17°. For a human-sized leg (~90 cm) that corresponds to roughly 7–8 cm roots and stones. The ground is flat where the robot starts and blends into the rough region ahead.

### What the controller adds

1. **Terrain-aware swing.** Each foot keeps its own ground height. The swing foot is lifted high enough to clear the highest point along its path and lands at the terrain height of its target.
2. **Slope-adapted foot orientation.** A least-squares plane $z = a\,\Delta X + b\,\Delta Y + c$ is fitted to the terrain under the footprint (normal equations $A^\top A\,[a\;b\;c]^\top = A^\top h$), the tilt is clipped to ±8°, and the plane is shifted up so that no sample lies above the sole. The ankle is commanded so the sole lies on that plane instead of resting on one edge.
3. **Early touchdown.** If the swing foot's contact force exceeds 25 % of the robot weight before the planned landing, the descent is frozen there. The new ground height is taken from the measured joint angles (forward kinematics), not from the command, because the real foot lags the command by 6–10 mm.
4. **Reach guard.** If a leg approaches full extension, the hip is lowered so the knee never locks and the IK never saturates.
5. **Plan re-anchoring.** The ZMP reference and the remaining footsteps follow where the foot actually landed, so the plan never runs ahead of the robot.
6. **Ankle balance loop.** The stance-ankle angles are shifted in proportion to the measured DCM error (2 rad per metre, clipped to ±0.1 rad), which moves the center of pressure towards the direction of the fall. The DCM error is measured against the true CoM of the commanded posture (see below).

Derivations and numerical examples for every step are in the methods note [`docs/rough_terrain_methods.pdf`](docs/rough_terrain_methods.pdf) (Turkish). A hands-on guide to reading, running and extending the code is in [`docs/code_guide_tr.pdf`](docs/code_guide_tr.pdf) (Turkish).

### Results

All runs: 24 steps of 3 cm at 0.9 s per step (0.69 m planned), MuJoCo 4.0.1; reproduce with `python tools/run_terrain_tests.py`. "Walked" means the run reached the end without falling.

| Experiment | Result |
| --- | --- |
| Flat, `--rough 0.3`, `--rough 0.6` | walked, 0.68 m |
| `--rough 1.0`, 6 different terrains | 6 / 6 walked, 0.64–0.67 m |
| `--rough 1.5`, 3 different terrains | 3 / 3 walked, 0.63–0.64 m |
| `--rough 2.0` | fell |
| `--rough 1.0 --blind` (no height map) | walked, 0.65 m |
| `--rough 1.0 --no-slope` (soles kept level) | walked, 0.58 m |
| Flat ground, 0.1 s push: lateral +25 N / −14 N, forward +12 N / −10 N | walked |
| Flat ground, 0.1 s push: lateral −17 N, forward −14 N | fell |
| `--rough 0.6`, 0.1 s push: lateral ±8 N, forward +6 N | walked |

**Effect of the ankle balance loop** (same cases with `--no-ankle`): the −14 N lateral push is survived only with the loop; on `--rough 1.5` the largest torso pitch drops from 9–19° to 5–9°; walking blind on `--rough 1.0` reaches 0.65 m instead of 0.58 m. At `--rough 1.0` and below both settings walk all cases. Single runs are sensitive to small numerical differences, so treat individual distances as ±2 cm.

**DCM foot placement does not help at this step time.** With DCM on and off the robot survived the same pushes, and on rough terrain DCM mostly reduced the distance walked, so it is off by default here. The reason is the time scale: for this robot $1/\omega = \sqrt{z_c/g} = \sqrt{0.185/9.81} = 0.137$ s, while one step takes 0.9 s. A DCM error grows as $e^{\omega t}$, i.e. about 27× in just half a step (0.45 s), so by the time the corrected foot lands it is too late. The flat-ground controller (`main.py`) uses 0.45 s steps, and there DCM does work (30 N push survived with DCM, fell without). Consistently, with `--step-time 0.45 --dcm` the terrain controller survives a −17 N push that it does not survive without DCM, but it covers only about half the planned distance; a faster terrain gait is future work.

### Lessons from the real meshes

The first version was tested with placeholder box meshes and appeared to work; with the real STL meshes it fell even on flat ground. The causes, all fixed and documented in `planners/terrain_planner.py`:

- **Wrong DCM reference.** The LIPM puts all the mass at the hip, but this robot's legs are heavy: the plan swings the hip ±71 mm while the true CoM swings only ±48 mm. Comparing the measured DCM with the LIPM state produced a 60 mm "error" with no push at all. The DCM error is now computed against the true CoM of the commanded posture.
- **ZMP reference not following the foot.** When a foot was moved, the ZMP reference for the next support phase stayed at the old spot.
- **Touchdown height from the command.** It is now taken from the measured joint angles.
- **Stopping onto the front foot.** When the walk ended, the ZMP reference went onto the front foot instead of between both feet.

### Limitations of the rough-terrain controller

- **Ground-truth terrain.** The controller reads the terrain height directly from the simulation. A real robot would need a depth camera or LiDAR plus elevation mapping; `--blind` shows what happens without it.
- **Short and slow.** Straight-ahead walking with 3 cm steps at 0.9 s per step.
- **Rigid ground, one simulator.** The soil is modeled as a slightly soft rigid contact, in MuJoCo only.

---

## Hardware Platform

The simulation was developed alongside a physical 12-DoF prototype. The planned hardware stack:

| Subsystem | Choice |
| --- | --- |
| **Processing** | Teensy 4.1 (500 Hz online stabilization loop) |
| **Actuation** | ST series serial bus servomotors (e.g., ST 3025) |
| **State estimation** | MPU6050 IMU for torso orientation |
| **Contact sensing** | Custom foot pads with FSR (force-sensing resistor) arrays to estimate ground reaction forces and the ZMP |
| **Mechanical design** | Custom 3D-printed or machined brackets matched to the mass and inertia values in the MuJoCo model |

![Prototype](assets/prototype.jpeg)

This repository contains the **Python simulation only**; the microcontroller firmware is not included. Full sim-to-real transfer of the push-recovery algorithm on the physical platform is future work.

---

## Limitations

Being explicit about what this project does *not* (yet) show:

- **Simulation only.** No results on the physical robot are reported.
- **LIPM assumptions.** The planner assumes a point-mass CoM at constant height and neglects leg dynamics. Real legs have mass and swing dynamics that the feedback loops must absorb.
- **Ideal-ish environment.** `main.py` walks on flat ground; `main_terrain.py` walks on uneven ground but reads the terrain height directly from the simulator. Actuator dynamics (servo lag, backlash, torque limits, bus latency) and sensor noise are not yet modeled in detail.
- **Limited push evaluation.** The documented push test is a 30 N, 0.1 s lateral push at a fixed time. The full recoverable envelope (force, direction, gait phase) is not yet characterized.
- **Single physics engine.** Results come from MuJoCo only; contact models differ between simulators.

---

## Roadmap

- [x] ZMP preview control (LIPM + DARE) trajectory generation
- [x] LQR-based online stabilizer
- [x] DCM push recovery with foot-landing adjustment
- [x] Analytical inverse kinematics (+ optional DLS refinement)
- [x] Push visualization, video recording, telemetry logging
- [x] Rough-terrain walking: terrain-aware swing, slope-adapted feet, early touchdown, ankle balance loop
- [x] Unit tests and continuous integration
- [ ] Faster terrain gait (0.45 s steps) where DCM foot placement becomes effective
- [ ] Perception-based height map (depth camera / elevation mapping) instead of simulator ground truth
- [ ] Systematic benchmark suite with randomized trials and confidence intervals
- [ ] Push-recovery envelope (force × direction × gait phase)
- [ ] Realistic actuator model (lag, saturation, backlash, communication delay)
- [ ] Sensor models and state estimation (IMU noise, FSR-based ZMP)
- [ ] Processor-in-the-loop test on a real Teensy 4.1
- [ ] Whole-body control (WBC) extension
- [ ] Firmware release and hardware sim-to-real validation

---

## Troubleshooting

| Problem | What to try |
| --- | --- |
| No window opens (remote server, WSL, CI) | Use `--headless`. |
| Video recording fails or is empty | Make sure `imageio` and `imageio-ffmpeg` are installed. On Linux without a display, set `MUJOCO_GL=egl` (or `osmesa`) before running. |
| Torso-tilt ankle feedback makes balance worse | Run with `--ankle-fb --ankle-fb-sign -1`. |
| You want to isolate the effect of DCM | Compare `--no-dcm` with the default, or sweep `--k-dcm` between 0 and 1. |
| Simulation runs too slowly or too fast | Adjust `--speed` (1.0 = real time, 0 = as fast as possible). |
| `Error opening file 'robot_walk.xml'` or `ModuleNotFoundError` | Update to the latest version: all paths are now resolved against the repository root, so any working folder is fine. Do not move single scripts out of the repository. |
| `main_terrain.py`: "height map and the MuJoCo heightfield differ" | The terrain written to MuJoCo differs from the controller's copy; open an issue, or run with `--blind` as a workaround. |
| The robot falls in `main_terrain.py` | Check the summary's torso tilt; try `--rough` one step lower, another `--seed`, or `--no-ankle` to compare. |

---

## Contributing

Issues and pull requests are welcome, especially for benchmarks, tests, actuator/sensor models and firmware ports. Please keep MuJoCo-specific code inside `sim/mujoco_interface.py` so the controllers stay portable.

---

## Citation

If you use this work, please cite the paper above:

```bibtex
@inproceedings{unver2026go2b,
  title     = {Development of Model Predictive Controlled 12-DoF Bipedal Robot: A Two-Phase Control Architecture Tailored for Low-Budget Bipedal Systems},
  author    = {{\"U}nver, Hakk{\i} {\"O}zg{\"u}r and Lotfi, B. and Eser, Giray Cevher and Ecevit, O.},
  booktitle = {21st International Conference on Machine Design and Production (UMTIK)},
  year      = {2026},
  address   = {Istanbul, T{\"u}rkiye}
}
```

---

## References

- S. Kajita et al., "The 3D Linear Inverted Pendulum Mode: A simple modeling for a biped walking pattern generation," *IROS*, 2001.
- S. Kajita et al., "Biped walking pattern generation by using preview control of zero-moment point," *ICRA*, 2003.
- M. Vukobratović and B. Borovac, "Zero-Moment Point — Thirty Five Years of Its Life," *International Journal of Humanoid Robotics*, 2004.
- J. Pratt et al., "Capture Point: A Step toward Humanoid Push Recovery," *IEEE-RAS Humanoids*, 2006.
- J. Englsberger, C. Ott, A. Albu-Schäffer, "Three-Dimensional Bipedal Walking Control Based on Divergent Component of Motion," *IEEE Transactions on Robotics*, 2015.
- E. Todorov, T. Erez, Y. Tassa, "MuJoCo: A physics engine for model-based control," *IROS*, 2012.

---

## License

Released under the [MIT License](LICENSE).
