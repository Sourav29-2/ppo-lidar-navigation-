# Roadmap

This project is actively maintained and designed to be extended.
Pull it to your robot, add your sensors, and build on it.

---

## Current Status

| Feature | Status |
|---|---|
| LiDAR-based PPO local navigation | ✅ Done |
| SLAM-based map building (SLAM Toolbox) | ✅ Done |
| Nav2 global path planner integration | ✅ Done |
| Hybrid Nav2 + PPO policy (50D obs) | ✅ Done |
| 5-phase curriculum training | ✅ Done |
| Interactive RViz click-to-navigate demo | ✅ Done |
| 82.7% success on 150 benchmark scenarios | ✅ Done |
| One-command setup (Mac + Windows WSL2) | ✅ Done |

---

## Planned Extensions

### 🤖 Real Robot Deployment
- [ ] Hardware interface node (ROS2 `hardware_interface`) for physical robot
- [ ] Velocity safety layer (enforce physical limits on real hardware)
- [ ] Localization tuning for real LIDAR vs simulation gap
- [ ] Field testing log and results

### 📷 Camera + Object Detection
- [ ] RGB-D camera integration (Intel RealSense or OAK-D)
- [ ] YOLOv8 object detection node
- [ ] Semantic observation extension: add detected object classes/distances to obs space
- [ ] Avoidance of dynamic obstacles (moving people)

### 🎙️ Voice Command Interface
- [ ] Whisper STT → room name → Nav2 goal publisher
- [ ] "Go to kitchen" / "Go to bedroom" voice commands
- [ ] Confirmation audio feedback

### 🗺️ Advanced Navigation
- [ ] Multi-room map with semantic room labels
- [ ] Multi-floor navigation (elevator integration)
- [ ] Dynamic obstacle avoidance (moving people via camera)

### 📊 Monitoring & Logging
- [ ] Prometheus + Grafana dashboard for live navigation metrics
- [ ] Automated regression testing on every training run

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on adding new sensors,
training on new environments, or extending the policy.

---

## Want to collaborate?

If you're working on mobile robotics, autonomous navigation, or RL-based control
and want to collaborate — reach out! Contact details in the README.
