# Hardware-in-the-loop harness (Isaac Sim side)

`hitl_sim.py` streams the simulated Traxxas's camera to a robot computer (`onboard/hitl_pi.py` on a real
Raspberry Pi 5 + Hailo-8L, or the same code on the laptop), applies each returned command only when it arrives, and logs
pose, commands and timing per step. `hitl_scene.py` places the props with colliders sized to what the car can hit
(`props_manifest.json`); `scenarios.json` holds the ten frozen scenes; `kino_plan.py` is the oracle planner.

Released for reference: it needs Isaac Sim / Isaac Lab 3.0 and the project's yard environment (`yard_v1_env_cfg`,
`drivable_map`, the Gaussian-splat scan), which are not released because the scan is of a private property. The per-step
logs of every evaluated arm are in the Hugging Face dataset (`hitl/runs.tar.zst`), and `bench/hitl_data.py` re-scores
them.

    ./run_headless.sh log.txt hitl_sim.py --brain pi --host <pi> --nav --slam --realtime --esc-gate \
        --cam-h 0.133 --stuck-s 35 --odo imu --seed 1
