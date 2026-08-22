# 12-DOF Bipedal Walking — Modular Architecture

```
config/parameters.py            RobotConfig, GaitConfig, DCMConfig, PushConfig,
                                AnkleFeedbackConfig, SimConfig   (tum sabitler)
kinematics/leg_ik.py            LegKinematics                    (Kajita IK + FK)
controllers/preview_controller.py  ZMPPreviewController          (LIPM + DARE)
controllers/push_recovery.py    DCMRecovery, AnkleStabilizer     (capture point)
planners/step_planner.py        StepPlanner                      (ayak izi + salinim + ZMP ref)
utils/data_logger.py            TelemetryLogger                  (log + ozet + grafik)
sim/mujoco_interface.py         MujocoRobot, VideoRecorder       (TEK mujoco bagimliligi)
tools/build_model.py            robot_final.xml -> robot_walk.xml
main.py                         orkestrasyon
```

`config`, `kinematics`, `controllers`, `planners`, `utils` **mujoco'ya bagimli
degildir** — saf numpy/scipy, mujoco kurulu olmadan test edilebilir.

## Kullanim

```bash
python tools/build_model.py --in robot_final.xml
python main.py
python main.py --headless --push 15
python main.py --record yuruyus.mp4 --headless --width 1920 --height 1080
python main.py --push 15 --push-viz ball --no-dcm
```

## Veri akisi (her dt_mpc = 10 ms)

```
StepPlanner.zmp_ref/preview_window
        |
        v
ZMPPreviewController.update()  x2 (yanal, ileri)  ->  planlanan CoM
        |
MujocoRobot.com_state()  ->  olculen CoM
        |
        v
DCMRecovery.compute()  ->  (d_lat, d_fwd)
        |
        v
StepPlanner.update(k, correction)  ->  FootTargets
        |
        v
LegKinematics.solve()  x2  ->  12 eklem acisi
        |
        v
MujocoRobot.send() -> step()
```

## Dogrulama (monolitik surumle karsilastirma)

```
IK eklem acisi farki        0.000e+00 rad   (130 hedef)
ZMP referans dizisi farki   0.000e+00
salinim lift profili farki  0.000e+00
preview kazanclari          Ks, Kx, F birebir ayni
```
