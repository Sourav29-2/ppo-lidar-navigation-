# Critical Environment Actions and Launch Commands

## 1. Launching Gazebo and RViz (Simulation Backend)
**NEVER** run the visual nodes inside the `PPO_lidar_navigation` environment, as it causes massive startup delays (3-4 minutes) and `libQt5Core`/`libQt6Core` casting conflicts due to overlapping Python/Pixi dependencies.

You MUST point to the `ros2_study` environment using the `--manifest-path` flag to launch it cleanly and instantly.

**Correct Command (Run as background task):**
```bash
pixi run --manifest-path /Users/souravkumar/ros2_study/pixi.toml bash -c "export KMP_DUPLICATE_LIB_OK=TRUE && export ROS_LOCALHOST_ONLY=1 && source /Users/souravkumar/ros2_study/install/setup.bash && ros2 launch urdf_test practise_gazebo.launch.py"
```

## 2. Launching Nav2 (Required for Goal Sampling)
Nav2 MUST be running in the background because the environment (`gazebo_nav_env.py`) queries the `/plan` topic to sample valid goals (goals not inside walls) and to calculate the dynamic `T_max`.

**Correct Command (Run as background task):**
```bash
pixi run --manifest-path /Users/souravkumar/ros2_study/pixi.toml bash -c "export KMP_DUPLICATE_LIB_OK=TRUE && export ROS_LOCALHOST_ONLY=1 && source /Users/souravkumar/ros2_study/install/setup.bash && ros2 launch urdf_test nav2.launch.py"
```

## 3. Launching the RL Training Script
The PyTorch/Stable-Baselines3 training script (`train_stage_a.py`) must be run inside the `PPO_lidar_navigation` environment where ML dependencies are installed. It still needs to source the `ros2_study` workspace to access the ROS 2 message types (`rclpy`, `nav2_msgs`, etc.). 

**CRITICAL BUG FIX**: Do NOT use `pixi run bash -c "..."` for this. It causes `_local_setup_util.py` to hang infinitely because it mixes the Pixi environment hooks with the ROS 2 python environment. Instead, use a pure bash subshell and call the absolute path to the local Pixi Python interpreter!

**Correct Command (Run as background task):**
```bash
bash -c "export KMP_DUPLICATE_LIB_OK=TRUE && export ROS_LOCALHOST_ONLY=1 && source /Users/souravkumar/ros2_study/install/setup.bash && /Users/souravkumar/Documents/Codex/2026-08-06/PPO_lidar_navigation/.pixi/envs/default/bin/python -u train_stage_a.py"
```

## Critical Rules to Prevent Future Errors:
1. **Never guess launch files:** The Gazebo launch file is strictly `practise_gazebo.launch.py`. Do not assume or hallucinate `test.launch.py` or similar names.
2. **Environment Isolation:** Use the `--manifest-path` flag when you need to trigger a command belonging entirely to the `ros2_study` workspace while your active directory is elsewhere.
3. **No Headless Modes:** Never run Gazebo headless, always let the GUI pop up for the user to monitor.
