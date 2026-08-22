# GO2B: 12-DOF Bipedal Technology Demonstrator
A high-fidelity bipedal robotics platform featuring ZMP-based trajectory planning and MuJoCo simulation.  

Özgür Hakkı Ünver, Bahram Lotfi , Ozan Ecevit , Giray Cevher Eser 





Institutional Acknowledgment
The GO2B Bipedal Project was developed and prototyped within the System Dynamics and Control Laboratory at TOBB University of Economics and Technology (TOBB ETÜ). The integration of complex control algorithms with physical hardware was conducted under the academic and technical standards supported by the university's research infrastructure.

Project Overview
GO2B is an R&D platform designed to validate control architectures for bipedal locomotion. This repository demonstrates the integration of MPC-based Preview Control, Inverse Kinematics (IK), and real-time sensor feedback within a MuJoCo physics environment. The project serves as a technology demonstrator for testing advanced stability algorithms on a 12-DOF physical prototype.

Key Capabilities
MPC Trajectory Planning: Robust locomotion planning using ZMP Preview Control and Linear Inverted Pendulum Model (LIPM).

Sim-to-Real Architecture: Python-based control loops optimized for hardware interfacing and real-time sensor data logging.

Mechanical Design: 12-DOF bipedal leg system, designed and optimized for precision CNC manufacturing.

GenAI Integration: Experimental autonomous visual route planning utilizing Robotics-E.R 1.6 Gemini API for adaptive navigation.

Technical Stack
Simulation: MuJoCo Physics Engine (robot_v2.xml)

Control Theory: ZMP Preview Control (LQR/MPC)

Kinematics: 12-DOF Inverse Kinematics (IK)
### Kinematic Configuration
![GO2B Robot Mechanism](assets/structrue.jpg)

Interfacing: Python (Simulation & Control Loops) / MATLAB (Trajectory Optimization)

Acknowledgements
Mathematical Framework: The core ZMP Preview Control framework is adapted from Eko Rudiawan's ZMP-Preview-Control-WPG repository.

Kinematics: The Inverse Kinematics (IK) implementation is based on the methods described in Introduction to Humanoid Robotics (Kajita, 2014).

System Integration: All physical modeling, system-level architecture, GenAI integration, and Sim-to-Real adaptation were developed exclusively for the GO2B project.

Current Limitations
As an active R&D platform, the current control architecture is optimized for flat-surface locomotion. Ongoing work focuses on:

Integrating higher-torque BLDC motor drivers for increased payload capacity.

Improving balance stability during non-periodic terrain perturbations (Push Recovery).
