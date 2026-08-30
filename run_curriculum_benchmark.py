import sys
import torch
import numpy as np
from pathlib import Path

from src.env.gazebo_mac_env import GazeboMacEnv
from src.rl.ppo import PPOActor
from src.rl.utils import evaluate_policy_detailed

from scratch.analyze_goal_difficulty import classify_goal_difficulty

BOX_OBSTACLES = [
    (0, 5, 10, 0.4),
    (0, -5, 10, 0.4),
    (5, 0, 0.4, 4),
    (-5, 0, 0.4, 4)
]
CYLINDER_OBSTACLES = [
    (3, 3, 0.5),
    (-3, -3, 0.5),
    (-3, 3, 0.5),
    (3, -3, 0.5)
]

def is_valid_goal(x, y, robot_pos, min_dist=1.5):
    if x < -7.5 or x > 7.5 or y < -7.5 or y > 7.5: return False
    if np.sqrt((x - robot_pos[0])**2 + (y - robot_pos[1])**2) < min_dist: return False
    
    for cx, cy, szx, szy in BOX_OBSTACLES:
        if (cx - szx/2 - 0.3) < x < (cx + szx/2 + 0.3) and (cy - szy/2 - 0.3) < y < (cy + szy/2 + 0.3):
            return False
            
    for cx, cy, r in CYLINDER_OBSTACLES:
        if np.sqrt((x - cx)**2 + (y - cy)**2) < (r + 0.3):
            return False
            
    return True

def generate_benchmark_dataset(rng, num_per_category=25):
    goals = []
    robot_start = np.array([0.0, 0.0])
    
    categories = ["EASY", "MODERATE", "WALL-BLOCKED", "COMPLEX"]
    for cat in categories:
        count = 0
        while count < num_per_category:
            x = rng.uniform(-7.5, 7.5)
            y = rng.uniform(-7.5, 7.5)
            if is_valid_goal(x, y, robot_start, min_dist=1.5):
                difficulty = classify_goal_difficulty(0.0, 0.0, x, y)
                if difficulty == cat:
                    goals.append((np.array([x, y], dtype=np.float32), cat))
                    count += 1
    return goals

def evaluate_model(env, model_path, goals, device):
    print(f"\\n--- Evaluating {model_path.name} ---")
    
    if not model_path.exists():
        print(f"Error: {model_path} not found.")
        return None
        
    actor = PPOActor(40, 2).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    if "actor_state_dict" in ckpt:
        actor.load_state_dict(ckpt["actor_state_dict"])
    elif "actor" in ckpt:
        actor.load_state_dict(ckpt["actor"])
    else:
        actor.load_state_dict(ckpt)
    
    res = evaluate_policy_detailed(env, actor, goals, device, num_episodes=len(goals))
    
    print(f"Success: {res['success']/res['total']*100:.1f}%")
    print(f"Collision: {res['collision']/res['total']*100:.1f}%")
    print(f"Timeout: {res['timeout']/res['total']*100:.1f}%")
    print(f"Stagnation Events: {res['stagnation_events']}")
    
    for k, v in res['corners'].items():
        if v['total'] > 0:
            print(f"  {k} Success: {v['success']/v['total']*100:.1f}% ({v['success']}/{v['total']})")
            
    return res

def main():
    device = torch.device("cpu")
    env = GazeboMacEnv()
    rng = np.random.default_rng(42)
    
    dataset = generate_benchmark_dataset(rng, num_per_category=25)
    print(f"Generated Benchmark Dataset with {len(dataset)} goals.")
    
    checkpoints_dir = Path("checkpoints")
    models = {
        "Champion": checkpoints_dir / "best_success.pt",
        "Oscillation": checkpoints_dir / "oscillation_latest.pt",
        "Curriculum": checkpoints_dir / "hybrid_curriculum_latest.pt"
    }
    
    all_results = {}
    try:
        for name, path in models.items():
            res = evaluate_model(env, path, dataset, device)
            all_results[name] = res
    finally:
        env.close()
        
    # Write report
    report_path = Path("diagnostics/curriculum_100k_report.txt")
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write("CURRICULUM BENCHMARK REPORT\\n")
        f.write("===========================\\n\\n")
        for name, res in all_results.items():
            if not res: continue
            f.write(f"Model: {name}\\n")
            f.write(f"Overall Success: {res['success']/res['total']*100:.1f}%\\n")
            f.write(f"Overall Collision: {res['collision']/res['total']*100:.1f}%\\n")
            f.write(f"Overall Timeout: {res['timeout']/res['total']*100:.1f}%\\n")
            f.write(f"Stagnation Events: {res['stagnation_events']}\\n")
            f.write("Breakdown:\\n")
            for k, v in res['corners'].items():
                if v['total'] > 0:
                    f.write(f"  {k}: {v['success']/v['total']*100:.1f}%\\n")
            f.write("\\n")
            
    print(f"\\nReport saved to {report_path}")

if __name__ == "__main__":
    main()
