# 12-DOF Bipedal Robot Simulation with ZMP & LQR Control

This repository contains the simulation and control architecture for a custom 12-Degree-of-Freedom (DOF) bipedal robot. The project is built in Python using MuJoCo for physics simulation and focuses on robust dynamic walking and balance recovery.

The software architecture is highly modular and designed to run on low-budget, resource-constrained hardware like the Teensy 4.1 microcontroller. To achieve this, the heavy computational tasks are separated from real-time stabilization.

## Academic Publication

The methodologies, control algorithms, and mathematical models in this repository are based on our peer-reviewed conference paper:

> **"Development of Model Predictive Controlled 12-DoF Bipedal Robot: A Two-Phase Control Architecture Tailored for Low-Budget Bipedal Systems"**  
> *Presented at the 21st International Conference on Machine Design and Production (UMTIK 2026), Istanbul, Türkiye.*  
> [Read the Full Paper Here](https://github.com/giraycevher/GO2B-Bipedal-Robot/blob/a86be1cf53217b836f31226f492656ad2d836d5f/assets/Development_of_Model_Predictive_Controlled_12-DoF_Bipedal_Robot.pdf)

## Key Features

* **Two-Phase Hierarchical Control:** 
  * **Offline:** Uses ZMP (Zero-Moment Point) Preview Control with a receding-horizon approach to generate stable walking trajectories.
  * **Online:** Uses an ultra-fast LQR (Linear Quadratic Regulator) for real-time balance correction.
* **DCM (Capture Point) Push Recovery:** The robot can withstand external impacts (e.g., a 30N push). It instantly calculates the state error of the Center of Mass (CoM) and actively changes its foot landing positions to maintain balance.
* **Analytical Inverse Kinematics (IK):** Instead of using computationally heavy Jacobian matrix solvers, this project uses a purely geometric, matrix-free IK solver tailored for 6-DOF legs.
* **Hardware-Ready Modular Design:** The physics engine (MuJoCo) is completely isolated from the control algorithms. You can run the planners and controllers on actual hardware without needing MuJoCo.

## Modular Architecture

The monolithic codebase has been refactored into a clean, ROS-inspired structure:

```text
├── config/parameters.py               # All constants (Gains, Limits, Sim params)
├── kinematics/leg_ik.py               # Analytical Inverse & Forward Kinematics
├── controllers/preview_controller.py  # LIPM and Discrete Algebraic Riccati Equation (DARE)
├── controllers/push_recovery.py       # DCM Capture Point logic and LQR stabilization
├── planners/step_planner.py           # Footstep trajectory and swing leg generation
├── utils/data_logger.py               # Real-time telemetry, logging, and plotting
├── sim/mujoco_interface.py            # The ONLY module dependent on MuJoCo
├── tools/build_model.py               # Utility to fix CAD-exported XML files
└── main.py                            # Main orchestration loop
