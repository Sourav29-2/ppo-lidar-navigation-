# Contributing

Thank you for your interest in extending this project!

---

## How to Add a New Sensor

### Example: Adding an RGB-D Camera

1. **Add the sensor to the URDF**  
   Edit `src/urdf_test/urdf/` — add the camera link and Gazebo plugin.

2. **Create a ROS2 subscriber in `gazebo_nav_env.py`**
   ```python
   from sensor_msgs.msg import Image
   self.camera_sub = self.node.create_subscription(
       Image, '/camera/color/image_raw', self._camera_cb, 10
   )
   ```

3. **Extend the observation space**  
   Add your new features to `_get_obs()` and update `OBS_DIM` accordingly.

4. **Update the Actor/Critic input dimension**
   ```python
   actor = Actor(obs_dim=NEW_DIM, action_dim=2)
   ```

5. **Retrain** from the existing champion checkpoint:
   ```bash
   python training/train_phase5_safety.py  # set CHAMPION_CKPT to your base
   ```

---

## How to Add a New Environment / World

1. Create a new `.world` file in `src/urdf_test/worlds/`
2. Create the corresponding map files in `src/urdf_test/maps/`
3. Update `BOX_OBSTACLES` in the training script to match your new world geometry
4. Generate a new benchmark scenarios JSON for evaluation

---

## Code Style

- Python: PEP 8, type hints where practical
- All new reward components must be logged to the `info` dict
- Training scripts: hard-stop at a fixed step count (no open-ended runs)
- Evaluation: always use the same 150-scenario benchmark for fair comparison

---

## Running Tests

```bash
# Import sanity checks
python -c "from ppo.actor import Actor; print('OK')"

# Reward logic tests
python -m pytest tests/ -v

# Quick smoke test (1 episode, no GPU needed)
python evaluation/eval_final_phase5.py --episodes 1
```

---

## Submitting Changes

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/camera-integration`
3. Commit with clear messages: `git commit -m "feat: add RGB-D camera observation"`
4. Open a pull request with a description of what changed and why
