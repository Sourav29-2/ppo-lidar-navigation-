# Windows Setup (via WSL2)

ROS 2 Humble + Gazebo Classic require Linux. On Windows, the standard approach is
**WSL2 (Windows Subsystem for Linux 2)** with Ubuntu 22.04 — this is what every
major ROS2 project recommends for Windows users.

---

## Step 1 — Install WSL2 + Ubuntu 22.04

Open **PowerShell as Administrator** and run:

```powershell
wsl --install -d Ubuntu-22.04
```

> Restart your PC when prompted.

After restarting, Ubuntu will open automatically. Create a username and password when asked.

---

## Step 2 — Inside the Ubuntu (WSL2) terminal

Copy-paste this **single command**:

```bash
curl -fsSL https://raw.githubusercontent.com/Sourav29-2/ppo-lidar-navigation-/master/setup/setup_wsl_inner.sh | bash
```

This will:
1. Install pixi (package manager)
2. Clone this repo
3. Install all ROS2 + PyTorch dependencies
4. Build the ROS2 package

---

## Step 3 — Run the Demo

After setup completes:

```bash
cd ppo-lidar-navigation-
./demo.sh
```

Then in the RViz window that opens, click **"Nav2 Goal"** and click anywhere on the map.

---

## Step 4 — Tips & Recovery During Demo

### 1. Opening a New Ubuntu Tab
When opening a new tab in Windows Terminal, it defaults to **PowerShell**. 
To open a new Ubuntu tab instead:
1. Click the small **downward arrow (v)** next to the `+` button in the tab bar.
2. Select **Ubuntu** from the dropdown menu.

### 2. Manual Human Override (Teleop)
If the robot gets stuck during a presentation, you can take manual control. Open a new Ubuntu tab (as shown above) and run:
```bash
cd ~/ppo-lidar-navigation-
pixi shell
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```
Use `I`, `J`, `K`, `L` to drive out of the corner. The AI will pause while you drive and instantly take over again when you stop pressing keys!

### 3. Clearing the Workspace Cache
If Gazebo or RViz starts acting glitchy, gets stuck on a white screen, or loads broken maps from previous runs, you can wipe the temporary ROS/Gazebo cache files to force a clean start:
```bash
rm -rf ~/.gazebo ~/.rviz2 ~/.ros/log
```

---

## Notes

- **Display (GUI)**: Windows 11 supports WSLg natively (Gazebo/RViz windows appear automatically).
  On Windows 10, install [VcXsrv](https://sourceforge.net/projects/vcxsrv/) and run it before launching.
- **Performance**: WSL2 is nearly native-speed on modern hardware. The simulation runs fine.
- **GPU**: Not required. The PPO inference runs on CPU.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `wsl: command not found` | Update Windows to 21H2 or later |
| Gazebo window doesn't open | On Win 10: start VcXsrv first, then run `export DISPLAY=:0` |
| `pixi: command not found` | Run `export PATH="$HOME/.pixi/bin:$PATH"` and retry |
| ROS2 nodes can't talk to each other | Run `export ROS_LOCALHOST_ONLY=1` (already set by pixi.toml) |
