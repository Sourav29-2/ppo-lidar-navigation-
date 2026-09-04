# Checkpoints

## What's in this directory

| File | Description |
|---|---|
| `hybrid_phase5/best_success.pt` | ✅ **Final champion model** — 82.7% success over 150 scenarios |

## Intermediate checkpoints (available in GitHub Releases)

All intermediate training checkpoints (~50 MB total) are available as a zip in
[GitHub Releases](https://github.com/Sourav29-2/ppo-lidar-navigation-/releases).

| Checkpoint | Phase | Success Rate |
|---|---|---|
| `best_success.pt` (PPO-only champion) | Stage D | ~52% |
| `hybrid_phase4/best_success.pt` | Hybrid Nav2+PPO Phase 4 | ~64.2% |
| `hybrid_phase5/best_success.pt` | Phase 5 Safety Fine-tune | **82.7%** ✅ |

## Loading a checkpoint

```python
import torch
from ppo.actor import Actor

actor = Actor(obs_dim=50, action_dim=2)
ckpt  = torch.load("checkpoints/hybrid_phase5/best_success.pt", map_location="cpu")
actor.load_state_dict(ckpt["actor_state_dict"])
actor.eval()
```
