# Project-Scoped Agent Rules

To ensure successful pairing and execution in this workspace:

1. **Launch Environment Alignment**:
   - Always run the core Gazebo, Nav2, and RViz simulation launch files from the **`/Users/souravkumar/ros2_study`** workspace environment. This workspace has a clean Qt5 configuration and avoids binary/Qt conflicts with `pytorch` and `stable-baselines3` which are present in `PPO_lidar_navigation`.
   - Never run visual nodes in the `PPO_lidar_navigation` env as it will crash with library load conflicts (exit code -6).

2. **Network/Local Communication Setup**:
   - When launching the ROS 2 simulation tasks, always prefix them with `export ROS_LOCALHOST_ONLY=1`. This aligns the ROS 2 network layer with the training environment configuration and enables the nodes to communicate.

3. **Background Services & Client Guarantees**:
   - Run the simulation backend and clients in the agent background shell as persistent tasks (`run_command`).
   - Do not use AppleScript or GUI automation scripts to control the user's personal terminal.

4. **No Headless Simulation & No Modifications to ros2_study**:
   - Do NOT run Gazebo in headless mode (do NOT pass `'gui': 'false'`). Keep Gazebo Classic GUI and RViz active.
   - Do NOT modify any files inside the `/Users/souravkumar/ros2_study` workspace. All project code and configuration changes must be made within the `PPO_lidar_navigation` workspace only.

5. **Fine-Tuning Evaluation and Logging Strategy**:
   - For any current or future fine-tuning experiment, ALWAYS evaluate and checkpoint the model at high-frequency intervals (every 5k to 10k steps) rather than large gaps (like 50k).
   - Maintain a dedicated, continuous markdown log (or task tracker) during the training process that records the success/collision metrics of every 5k/10k evaluation so we can strictly verify step-by-step progress and catch regressions early.

6. **"Kill All" = Current Experiment Session Cleanup**:
   - Whenever the user issues the command "kill all", it MUST be strictly interpreted as a request to terminate all running processes, background tasks, persistent agent tasks, and sessions that belong to the CURRENT training, evaluation, or simulation experiment.
   - The purpose is to leave the environment clean so that the next experiment starts without interference from leftover processes.
   - **What "Kill All" May Terminate (if they belong to the CURRENT experiment)**: Gazebo, RViz, Nav2, SLAM, ROS 2 nodes, PPO training/evaluation processes, Python scripts, experiment monitoring/logging processes, background agent tasks.
   - **What "Kill All" Must NEVER Delete**: "KILL ALL" means STOP PROCESSES, NOT DELETE PROJECT RESOURCES. Never interpret "kill all" as permission to delete files, directories, source code, remove Python imports, uninstall libraries, remove configurations, model checkpoints, logs, datasets, or diagnostics. NEVER use "kill all" to execute destructive filesystem commands (e.g., `rm`, `pip uninstall`).
   - **Do Not Use Blind System-Wide Kills**: Do not blindly execute `killall` or `pkill` against broad process names (like `killall python`). Instead, identify processes belonging to the current experiment and terminate only those.
   - **Track Background Tasks**: Maintain enough information (experiment name, process ID, task ID, shell identity) for any background persistent task so it can be safely targeted later.
   - **Training/Evaluation/Simulation Cleanup**: Terminate the relevant running processes cleanly but PRESERVE all checkpoints, logs, directories, diagnostics, trajectories, etc. Verify that the simulation processes are gone without modifying `ros2_study`.
   - **"Kill" vs "Kill All"**: 
     - "kill Gazebo" -> Stop Gazebo only.
     - "kill RViz" -> Stop RViz only.
     - "kill training/evaluation" -> Stop the relevant process only.
     - "kill all" -> Stop ALL running processes/tasks belonging to the current experiment session.
   - **After "Kill All"**: Verify relevant processes stopped, verify no training/evaluation remains, verify simulation and agent tasks stopped. Do NOT delete files, uninstall anything, or modify `ros2_study`. The environment should be clean with no leftover processes and all files intact.
   - **Final Safety Rule**: Before executing any destructive filesystem/package operation, require an explicit user request. "Kill all" alone is NEVER sufficient authorization for deletion or uninstallation. Default action is always: STOP PROCESSES, PRESERVE FILES.
