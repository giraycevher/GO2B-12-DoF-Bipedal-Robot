# GO2B: 12-DOF Bipedal Robot Simulation with ZMP Preview Control, DCM Push Recovery & LQR

This repository contains the simulation and control software for **GO2B**, a custom 12-degree-of-freedom (DOF) bipedal robot. It is written in Python and uses [MuJoCo](https://mujoco.org/) for physics simulation. The project focuses on dynamic walking and push recovery.

The software is modular and designed for low-budget, resource-constrained hardware such as the Teensy 4.1 microcontroller: the heavy computation (ZMP trajectory generation) is separated from the real-time stabilization loop.

![GO2B robot](assets/v2_robot.png)

## Academic Publication

The methods in this repository are based on our conference paper:

> Ünver, H. Ö., Lotfi, B., Eser, G. C., Ecevit, O. (2026). **"Development of Model Predictive Controlled 12-DoF Bipedal Robot: A Two-Phase Control Architecture Tailored for Low-Budget Bipedal Systems."**
> *21st International Conference on Machine Design and Production (UMTIK 2026), Istanbul, Türkiye.*
> [Read the paper (PDF)](assets/Development_of_Model_Predictive_Controlled_12-DoF_Bipedal_Robot.pdf)

## Key Features

- **Two-phase hierarchical control**
  - **Offline:** ZMP (Zero-Moment Point) preview control (LIPM + discrete algebraic Riccati equation) with a receding horizon generates stable walking trajectories.
  - **Online:** a fast LQR-based stabilizer corrects balance in real time.
- **DCM (Capture Point) push recovery:** the robot withstands external pushes (a 30 N lateral push in our tests). It computes the divergent component of motion (DCM) error and adjusts the foot landing position to keep its balance. An optional torso-tilt to ankle feedback is also available.
- **Analytical inverse kinematics:** a geometric, matrix-free solver for 6-DOF legs instead of a Jacobian-based solver. An optional damped-least-squares refinement is available with `--refine-ik`.
- **Modular design:** controllers, planner and kinematics are separate packages. By design, only `sim/mujoco_interface.py` (plus the viewer and logging glue in `main.py`) touches MuJoCo.

## Repository Structure

```
├── main.py                            # Main orchestration loop and command-line interface
├── requirements.txt                   # Python dependencies
├── robot_final.xml, robot_walk.xml    # MuJoCo robot models
├── config/parameters.py               # All constants (gains, limits, simulation parameters)
├── kinematics/leg_ik.py               # Analytical inverse and forward kinematics
├── controllers/preview_controller.py  # LIPM and discrete algebraic Riccati equation (DARE) preview controller
├── controllers/push_recovery.py       # DCM capture-point logic and stabilization
├── planners/step_planner.py           # Footstep trajectory and swing-leg generation
├── utils/data_logger.py               # Telemetry, logging and plotting
├── sim/mujoco_interface.py            # MuJoCo wrapper (robot, video recorder, push visualization)
├── tools/build_model.py               # Utility to fix CAD-exported XML files
└── assets/                            # STL meshes for the XML models, images and media
```

## Installation

Python 3.9 or newer is recommended.

```bash
git clone https://github.com/giraycevher/GO2B-12-DoF-Bipedal-Robot.git
cd GO2B-12-DoF-Bipedal-Robot
pip install -r requirements.txt
```

`requirements.txt` installs `numpy`, `scipy`, `mujoco`, `matplotlib`, and `imageio` with `imageio-ffmpeg` (the last two are only needed for video recording).

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

Run `python main.py --help` for the full list.

### Outputs

- A one-line status report per simulated second in the terminal (CoM progress, hip height, swing leg, contacts, DCM error and correction).
- A telemetry plot (unless `--no-plot` is set).
- `main_walk_trace_push<F>n.csv`: raw trace with the time, 12 joint commands and actual joint angles, base pose, CoM position and velocity, and foot contacts.
- `main_walk_trace_push<F>n.metadata.json`: run settings, exit reason, software versions, and a walked/fell result.

## Simulation Results

We compare two scenarios.

**Scenario 1: walking without disturbance.** The robot tracked the offline-generated CoM trajectory on flat ground. Although the LIPM assumption neglects leg masses, the simulation showed stable heel-strike and swing-leg transients.

![Scenario 1](https://github.com/user-attachments/assets/616be702-e658-4a58-8e95-aec3bd1930fb)

![Scenario 1 graph](assets/result1.png)

**Scenario 2: 30 N lateral push.** A lateral force of 30 N was applied to the torso for 0.1 s during the walking cycle. The controller detected the drift. Instead of over-torquing the ankles, it moved the landing position of the swing leg toward the direction of the fall (capture point strategy).

![Scenario 2](https://github.com/user-attachments/assets/777ee1dc-1aa6-4a46-839b-686c971681ad)

![Scenario 2 graph](assets/result2.png)

## Hardware Platform

The simulation was developed alongside a physical 12-DOF prototype. The planned hardware stack:

- **Processing:** Teensy 4.1 (500 Hz online stabilization loop)
- **Actuation:** ST series serial bus servomotors (e.g., ST 3025)
- **State estimation:** MPU6050 IMU for torso orientation
- **Contact sensing:** custom foot pads with FSR (force-sensing resistor) arrays to estimate ground reaction forces and the ZMP
- **Mechanical design:** custom 3D-printed or machined brackets matched to the mass and inertia values in the MuJoCo model

![Prototype](assets/prototype.jpeg)

This repository contains the **Python simulation only**; the microcontroller firmware is not included. Full sim-to-real transfer of the push-recovery algorithm on the physical platform is future work.

## Citation

If you use this work, please cite the paper above.

## License

Released under the [MIT License](LICENSE).
