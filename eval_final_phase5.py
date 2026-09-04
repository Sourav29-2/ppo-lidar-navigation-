"""Final Validation Evaluation — Phase 5 Safety-Margin Hybrid PPO.

Rules:
  • DO NOT TRAIN. Inference-only.
  • Use best_success.pt from hybrid_phase5.
  • Run all 150 scenarios from diagnostics/benchmark_150_scenarios.json.
  • Strict goal threshold: distance_to_goal <= 0.25 m = SUCCESS.
  • Diagnose every failed episode.
  • Save:
      diagnostics/final_evaluation.csv
      diagnostics/final_failure_diagnostics.csv
      diagnostics/final_evaluation_report.txt
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")

import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "urdf_test" / "src" / "scripts"))

from gazebo_nav_env import GazeboMacNavEnv
from rl.actor import Actor

# ─── Constants ────────────────────────────────────────────────────────────────
STRICT_SUCCESS_DIST   = 0.25          # m — ONLY criterion used in this eval
ENV_SUCCESS_DIST      = 0.50          # m — environment internal threshold (unchanged)
MAX_STEPS_PER_EPISODE = 800

CHECKPOINT           = PROJECT_ROOT / "checkpoints" / "hybrid_phase5" / "best_success.pt"
SCENARIOS_FILE       = PROJECT_ROOT / "diagnostics" / "benchmark_150_scenarios.json"
OUT_EVAL_CSV         = PROJECT_ROOT / "diagnostics" / "final_evaluation.csv"
OUT_FAILURE_CSV      = PROJECT_ROOT / "diagnostics" / "final_failure_diagnostics.csv"
OUT_REPORT_TXT       = PROJECT_ROOT / "diagnostics" / "final_evaluation_report.txt"

# ─── Previous model comparison baselines ──────────────────────────────────────
BASELINE_CHAMPION = {
    "model": "Original Champion (PPO-only)",
    "total_success_pct": 52.0,
    "total_collision_pct": 32.7,
    "total_timeout_pct": 15.3,
    "wall_blocked_success_pct": 43.3,
    "complex_success_pct": 26.7,
    "open_success_pct": 86.7,
}
BASELINE_HYBRID = {
    "model": "Hybrid Phase 4 Champion (Nav2+PPO, 0.25m criterion)",
    "total_success_pct": 64.2,
    "total_collision_pct": 26.3,
    "total_timeout_pct": 9.5,
    "wall_blocked_success_pct": None,     # not separately tracked at strict 0.25m
    "complex_success_pct": None,
    "open_success_pct": None,
}

# ─── Failure root-cause labels ────────────────────────────────────────────────
FAILURE_REASONS = [
    "LATE_OBSTACLE_REACTION",
    "EXCESSIVE_SPEED",
    "PATH_DEVIATION",
    "FAILED_PATH_FOLLOWING",
    "UNSAFE_NAV2_PATH",
    "WRONG_LOCAL_MANEUVER",
    "TIMEOUT_LOW_PROGRESS",
    "OSCILLATION",
    "OTHER",
]


@dataclass
class StepTelemetry:
    step_idx: int = 0
    min_lidar: float = 0.0
    front_clear: float = 0.0
    left_clear: float = 0.0
    right_clear: float = 0.0
    linear_vel: float = 0.0
    angular_vel: float = 0.0
    ppo_linear: float = 0.0
    ppo_angular: float = 0.0
    dist_to_goal: float = 0.0
    robot_x: float = 0.0
    robot_y: float = 0.0


@dataclass
class EpisodeRecord:
    scenario_id: str = ""
    category: str = ""
    outcome: str = ""               # SUCCESS / COLLISION / TIMEOUT
    final_dist_to_goal: float = 0.0
    min_clearance: float = 9.9
    path_length: float = 0.0
    total_steps: int = 0
    mean_linear_vel: float = 0.0
    mean_angular_vel: float = 0.0
    max_linear_vel: float = 0.0
    oscillation_detected: bool = False
    no_turn_detected: bool = False
    fallback_used_count: int = 0
    failure_reason: str = ""        # populated only on failure
    failure_evidence: str = ""
    trajectory: list[StepTelemetry] = field(default_factory=list)


# ─── TurtleBotEnv wrapper ─────────────────────────────────────────────────────
class TurtleBotEnv(GazeboMacNavEnv):
    def reset(self, seed=None, options=None):
        if options is not None and "target_position" in options:
            orig = self.safe_goals
            self.safe_goals = [np.array(options["target_position"])]
            obs, info = super().reset(seed=seed, options=options)
            self.safe_goals = orig
            return obs, info
        return super().reset(seed=seed, options=options)


# ─── Sector extraction from 360-ray lidar obs ─────────────────────────────────
def extract_sectors(env) -> tuple[float, float, float, float]:
    """Return (min, front, left, right) clearance from current raw lidar."""
    try:
        lidar = np.array(env.laser_ranges, dtype=np.float32)
        n = len(lidar)
        front_idx = slice(0, max(1, n // 12))            # ~30° front
        left_idx  = slice(n // 4 - n // 12, n // 4 + n // 12)
        right_idx = slice(3 * n // 4 - n // 12, 3 * n // 4 + n // 12)
        return (
            float(np.min(lidar)),
            float(np.min(lidar[front_idx])),
            float(np.min(lidar[left_idx])),
            float(np.min(lidar[right_idx])),
        )
    except Exception:
        return (0.3, 0.3, 0.3, 0.3)


# ─── Oscillation detector ─────────────────────────────────────────────────────
def detect_oscillation(angular_history: list[float],
                        window: int = 20, threshold: float = 1.0) -> bool:
    if len(angular_history) < window:
        return False
    for i in range(len(angular_history) - window):
        seg = angular_history[i:i + window]
        pos = sum(1 for v in seg if v > threshold)
        neg = sum(1 for v in seg if v < -threshold)
        if pos >= 3 and neg >= 3:
            return True
    return False


# ─── Root cause classifier ────────────────────────────────────────────────────
def classify_failure(record: EpisodeRecord) -> tuple[str, str]:
    """Classify failure using trajectory evidence. Returns (reason, evidence)."""
    traj = record.trajectory
    if not traj:
        return "OTHER", "No trajectory data."

    # Extract last 10–20 steps for collision analysis
    analysis_window = traj[-min(20, len(traj)):]

    min_c_last = min(s.min_lidar for s in analysis_window)
    mean_lin   = np.mean([s.linear_vel for s in analysis_window])
    mean_ang   = np.mean([abs(s.angular_vel) for s in analysis_window])
    dist_start = analysis_window[0].dist_to_goal
    dist_end   = analysis_window[-1].dist_to_goal

    # ── TIMEOUT / LOW PROGRESS ───────────────────────────────────────────────
    if record.outcome == "TIMEOUT":
        low_progress = abs(dist_start - dist_end) < 0.5
        if record.oscillation_detected:
            return "OSCILLATION", (
                f"Timeout with oscillation. Δdist={abs(dist_start-dist_end):.2f}m | "
                f"mean|ω|={mean_ang:.3f} rad/s"
            )
        if record.no_turn_detected:
            return "FAILED_PATH_FOLLOWING", (
                f"Timeout, no meaningful turn detected. "
                f"max_ang={max((abs(s.angular_vel) for s in traj), default=0):.3f}"
            )
        if low_progress:
            return "TIMEOUT_LOW_PROGRESS", (
                f"Timeout with minimal progress. Δdist={abs(dist_start-dist_end):.2f}m | "
                f"final_dist={record.final_dist_to_goal:.2f}m"
            )
        return "TIMEOUT_LOW_PROGRESS", (
            f"Timeout. final_dist={record.final_dist_to_goal:.2f}m | "
            f"mean_vel={record.mean_linear_vel:.3f} m/s"
        )

    # ── COLLISION ─────────────────────────────────────────────────────────────
    if record.outcome == "COLLISION":
        # Oscillation → wrong maneuver
        if record.oscillation_detected:
            return "OSCILLATION", (
                f"Collision with oscillation in final {len(analysis_window)} steps. "
                f"min_clear={min_c_last:.3f}m | mean_lin={mean_lin:.3f}"
            )

        # High speed + dropping clearance = late reaction
        high_speed   = mean_lin > 0.12
        low_clearance = min_c_last < 0.35
        if high_speed and low_clearance:
            return "LATE_OBSTACLE_REACTION", (
                f"High forward velocity ({mean_lin:.3f} m/s) with dropping clearance "
                f"({min_c_last:.3f}m) in final {len(analysis_window)} steps."
            )

        # Very fast approach
        if record.max_linear_vel > 0.30 and low_clearance:
            return "EXCESSIVE_SPEED", (
                f"Max linear vel={record.max_linear_vel:.3f} m/s during approach. "
                f"min_clearance_last20={min_c_last:.3f}m"
            )

        # No turn and crashed → failed to follow path
        if record.no_turn_detected:
            return "FAILED_PATH_FOLLOWING", (
                f"No meaningful turn before collision. "
                f"max_angular={max((abs(s.angular_vel) for s in traj), default=0):.3f} | "
                f"min_clearance={record.min_clearance:.3f}m"
            )

        # Low clearance + low angular → wrong local maneuver
        if low_clearance and mean_ang < 0.15:
            return "WRONG_LOCAL_MANEUVER", (
                f"Low clearance ({min_c_last:.3f}m) with low angular response "
                f"({mean_ang:.3f} rad/s) — failed to turn away from obstacle."
            )

        # If still moving forward near a wall with moderate clearance
        if mean_lin > 0.05 and min_c_last < 0.50:
            return "LATE_OBSTACLE_REACTION", (
                f"Forward motion ({mean_lin:.3f} m/s) persisted near obstacle "
                f"(clearance={min_c_last:.3f}m). Late reaction."
            )

        return "OTHER", (
            f"Collision unclassified. min_clear={record.min_clearance:.3f}m | "
            f"mean_lin={mean_lin:.3f} | mean_ang={mean_ang:.3f}"
        )

    return "OTHER", f"Outcome={record.outcome} unexpectedly reached classifier."


# ─── Main evaluation ──────────────────────────────────────────────────────────
def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("FINAL HYBRID PPO VALIDATION — PHASE 5 SAFETY MODEL")
    print("=" * 60)
    print(f"  Checkpoint         : {CHECKPOINT}")
    print(f"  Scenarios          : {SCENARIOS_FILE}")
    print(f"  Strict success threshold : distance_to_goal <= {STRICT_SUCCESS_DIST} m")
    print(f"  Env internal threshold   : {ENV_SUCCESS_DIST} m (UNCHANGED)")
    print(f"  Max steps/episode  : {MAX_STEPS_PER_EPISODE}")
    print("=" * 60 + "\n")

    # ── Load scenarios ────────────────────────────────────────────────────────
    with open(SCENARIOS_FILE) as f:
        scenarios = json.load(f)
    print(f"  Loaded {len(scenarios)} scenarios.\n")

    # ── Build env ─────────────────────────────────────────────────────────────
    env = TurtleBotEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(f"  Obs dim: {obs_dim}  Act dim: {act_dim}\n")

    # ── Load actor ────────────────────────────────────────────────────────────
    actor  = Actor(observation_dim=obs_dim, action_dim=act_dim, hidden_sizes=(256, 256)).to(device)
    actor.eval()

    # Load actor weights directly from checkpoint (inference-only — no optimizer needed)
    # weights_only=False required: checkpoint was saved with numpy scalars embedded
    ckpt_data = torch.load(str(CHECKPOINT), map_location=device, weights_only=False)
    actor.load_state_dict(ckpt_data["actor"])
    print(f"  Actor loaded from: {CHECKPOINT.name}")
    print(f"  Checkpoint step: {ckpt_data.get('total_steps', 'unknown')}")
    print(f"  Best success rate in training: {ckpt_data.get('best_success_rate', 'unknown')}\n")

    # ── Run episodes ──────────────────────────────────────────────────────────
    records: list[EpisodeRecord] = []

    for sc_idx, sc in enumerate(scenarios):
        scenario_id = sc["scenario_id"]
        category    = sc.get("scenario_category", "UNKNOWN")
        goal_pos    = np.array(sc["goal_position"], dtype=np.float32)
        start_pos   = np.array(sc.get("start_position", [0.0, 0.0]), dtype=np.float32)

        rec = EpisodeRecord(scenario_id=scenario_id, category=category)

        obs, _ = env.reset(options={"target_position": goal_pos})
        step_idx = 0
        last_pos = env.robot_pos.copy()
        angular_history: list[float] = []
        linear_history:  list[float] = []
        min_clearance_seen = 9.9
        fallback_count = 0
        final_info: dict = {}
        strict_success = False
        strict_collision = False

        while step_idx < MAX_STEPS_PER_EPISODE:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                dist_ = actor(obs_t)
                action = torch.clamp(dist_.mean, -1.0, 1.0).cpu().numpy()[0]

            obs, _, terminated, truncated, info = env.step(action)
            final_info = info
            step_idx += 1

            # ── Telemetry ────────────────────────────────────────────────────
            min_c, front_c, left_c, right_c = extract_sectors(env)
            lin_vel = float(getattr(env, "current_linear_vel", 0.0))
            ang_vel = float(getattr(env, "current_angular_vel", 0.0))
            cur_pos = env.robot_pos.copy()
            d2g = float(np.linalg.norm(cur_pos - goal_pos))

            rec.path_length += float(np.linalg.norm(cur_pos - last_pos))
            last_pos = cur_pos.copy()
            min_clearance_seen = min(min_clearance_seen, min_c)
            angular_history.append(ang_vel)
            linear_history.append(lin_vel)

            # Track fallback usage via env info
            if info.get("path_stale", False):
                fallback_count += 1

            telem = StepTelemetry(
                step_idx=step_idx,
                min_lidar=min_c,
                front_clear=front_c,
                left_clear=left_c,
                right_clear=right_c,
                linear_vel=lin_vel,
                angular_vel=ang_vel,
                ppo_linear=float(action[0]),
                ppo_angular=float(action[1]),
                dist_to_goal=d2g,
                robot_x=float(cur_pos[0]),
                robot_y=float(cur_pos[1]),
            )
            rec.trajectory.append(telem)

            # ── Strict 0.25 m success criterion (checked every step) ──────────
            if d2g <= STRICT_SUCCESS_DIST:
                strict_success = True
                break

            # ── Collision is always terminal ──────────────────────────────────
            if bool(info.get("is_collision", False)):
                strict_collision = True
                break

            # ── Do NOT stop on env's 0.5m success — continue to reach 0.25m ──
            # Only truncated (step limit from env side) terminates the loop
            if truncated:
                break

        # ── Post-episode metrics ──────────────────────────────────────────────
        rec.total_steps = step_idx
        rec.min_clearance = min_clearance_seen
        rec.mean_linear_vel  = float(np.mean(linear_history)) if linear_history else 0.0
        rec.mean_angular_vel = float(np.mean([abs(v) for v in angular_history])) if angular_history else 0.0
        rec.max_linear_vel   = float(max(linear_history, default=0.0))
        rec.oscillation_detected = detect_oscillation(angular_history)
        rec.no_turn_detected = max((abs(v) for v in angular_history), default=0.0) < 0.3
        rec.fallback_used_count = fallback_count

        # ── Final distance and outcome ─────────────────────────────────────────
        final_pos = env.robot_pos.copy()
        final_dist = float(np.linalg.norm(final_pos - goal_pos))
        rec.final_dist_to_goal = final_dist

        if strict_success:
            rec.outcome = "SUCCESS"
        elif strict_collision:
            rec.outcome = "COLLISION"
        else:
            rec.outcome = "TIMEOUT"

        # ── Classify failures ─────────────────────────────────────────────────
        if rec.outcome != "SUCCESS":
            rec.failure_reason, rec.failure_evidence = classify_failure(rec)

        status_str = rec.outcome
        print(
            f"  [{sc_idx+1:3d}/{len(scenarios)}] {scenario_id:30s} | {status_str:9s} | "
            f"dist={final_dist:.3f}m | steps={step_idx} | "
            f"clear={rec.min_clearance:.3f}m | "
            f"{'FAIL:'+rec.failure_reason if rec.outcome!='SUCCESS' else ''}",
            flush=True,
        )

        records.append(rec)

    env.close()

    # ─── Aggregate results ────────────────────────────────────────────────────
    total = len(records)
    successes  = [r for r in records if r.outcome == "SUCCESS"]
    collisions = [r for r in records if r.outcome == "COLLISION"]
    timeouts   = [r for r in records if r.outcome == "TIMEOUT"]
    failures   = [r for r in records if r.outcome != "SUCCESS"]

    # By category
    cat_map: dict[str, list[EpisodeRecord]] = {}
    for r in records:
        cat_map.setdefault(r.category, []).append(r)

    def cat_stats(recs: list[EpisodeRecord]) -> dict:
        n = len(recs)
        s = sum(1 for r in recs if r.outcome == "SUCCESS")
        c = sum(1 for r in recs if r.outcome == "COLLISION")
        t = sum(1 for r in recs if r.outcome == "TIMEOUT")
        return {"n": n, "s": s, "c": c, "t": t,
                "s_pct": s/n*100 if n else 0,
                "c_pct": c/n*100 if n else 0,
                "t_pct": t/n*100 if n else 0}

    # Failure breakdown
    failure_counts: dict[str, int] = {}
    for r in failures:
        failure_counts[r.failure_reason] = failure_counts.get(r.failure_reason, 0) + 1

    # ─── Save final_evaluation.csv ────────────────────────────────────────────
    eval_cols = [
        "scenario_id", "category", "outcome", "final_dist_to_goal",
        "strict_success_025m", "min_clearance", "path_length", "total_steps",
        "mean_linear_vel", "mean_angular_vel", "max_linear_vel",
        "oscillation_detected", "no_turn_detected", "fallback_used_count",
        "failure_reason",
    ]
    with open(OUT_EVAL_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=eval_cols)
        w.writeheader()
        for r in records:
            w.writerow({
                "scenario_id":        r.scenario_id,
                "category":           r.category,
                "outcome":            r.outcome,
                "final_dist_to_goal": round(r.final_dist_to_goal, 4),
                "strict_success_025m": r.outcome == "SUCCESS",
                "min_clearance":      round(r.min_clearance, 4),
                "path_length":        round(r.path_length, 3),
                "total_steps":        r.total_steps,
                "mean_linear_vel":    round(r.mean_linear_vel, 4),
                "mean_angular_vel":   round(r.mean_angular_vel, 4),
                "max_linear_vel":     round(r.max_linear_vel, 4),
                "oscillation_detected": r.oscillation_detected,
                "no_turn_detected":   r.no_turn_detected,
                "fallback_used_count": r.fallback_used_count,
                "failure_reason":     r.failure_reason,
            })
    print(f"\n  [SAVED] {OUT_EVAL_CSV}")

    # ─── Save final_failure_diagnostics.csv ───────────────────────────────────
    fail_cols = [
        "scenario_id", "category", "outcome", "failure_reason",
        "final_dist_to_goal", "min_clearance", "path_length", "total_steps",
        "mean_linear_vel", "mean_angular_vel", "max_linear_vel",
        "oscillation_detected", "no_turn_detected", "fallback_used_count",
        "failure_evidence",
        # Final 20-step window summary
        "last20_min_lidar", "last20_mean_linear", "last20_mean_angular",
    ]
    with open(OUT_FAILURE_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fail_cols)
        w.writeheader()
        for r in failures:
            last20 = r.trajectory[-min(20, len(r.trajectory)):]
            w.writerow({
                "scenario_id":        r.scenario_id,
                "category":           r.category,
                "outcome":            r.outcome,
                "failure_reason":     r.failure_reason,
                "final_dist_to_goal": round(r.final_dist_to_goal, 4),
                "min_clearance":      round(r.min_clearance, 4),
                "path_length":        round(r.path_length, 3),
                "total_steps":        r.total_steps,
                "mean_linear_vel":    round(r.mean_linear_vel, 4),
                "mean_angular_vel":   round(r.mean_angular_vel, 4),
                "max_linear_vel":     round(r.max_linear_vel, 4),
                "oscillation_detected": r.oscillation_detected,
                "no_turn_detected":   r.no_turn_detected,
                "fallback_used_count": r.fallback_used_count,
                "failure_evidence":   r.failure_evidence,
                "last20_min_lidar":   round(min((s.min_lidar for s in last20), default=0.3), 4),
                "last20_mean_linear": round(float(np.mean([s.linear_vel for s in last20])) if last20 else 0.0, 4),
                "last20_mean_angular":round(float(np.mean([abs(s.angular_vel) for s in last20])) if last20 else 0.0, 4),
            })
    print(f"  [SAVED] {OUT_FAILURE_CSV}")

    # ─── Write final_evaluation_report.txt ────────────────────────────────────
    lines: list[str] = []
    def p(*args): lines.append(" ".join(str(a) for a in args))

    p("=" * 60)
    p("FINAL HYBRID PPO VALIDATION REPORT — PHASE 5 SAFETY MODEL")
    p("=" * 60)
    p()
    p(f"Checkpoint   : {CHECKPOINT}")
    p(f"Scenarios    : {total} (from benchmark_150_scenarios.json)")
    p(f"Strict threshold : distance_to_goal <= {STRICT_SUCCESS_DIST} m")
    p(f"Env threshold    : {ENV_SUCCESS_DIST} m (internal, unchanged)")
    p()

    # Section 1: Final Performance
    p("=" * 60)
    p("1. FINAL PERFORMANCE")
    p("=" * 60)
    p(f"  Total episodes : {total}")
    p(f"  SUCCESS        : {len(successes):3d}  ({len(successes)/total*100:.1f}%)")
    p(f"  COLLISION      : {len(collisions):3d}  ({len(collisions)/total*100:.1f}%)")
    p(f"  TIMEOUT        : {len(timeouts):3d}  ({len(timeouts)/total*100:.1f}%)")
    p()
    p("  Per-category breakdown:")
    p(f"  {'Category':<24} {'N':>4} {'Success%':>9} {'Collision%':>11} {'Timeout%':>9}")
    p("  " + "-" * 62)
    for cat in sorted(cat_map.keys()):
        st = cat_stats(cat_map[cat])
        p(f"  {cat:<24} {st['n']:>4} {st['s_pct']:>9.1f}% {st['c_pct']:>11.1f}% {st['t_pct']:>9.1f}%")
    p()
    p(f"  Mean min clearance (all) : {np.mean([r.min_clearance for r in records]):.3f} m")
    p(f"  Mean min clearance (success) : {np.mean([r.min_clearance for r in successes]):.3f} m" if successes else "")
    p(f"  Mean path length         : {np.mean([r.path_length for r in records]):.2f} m")
    p(f"  Mean episode steps       : {np.mean([r.total_steps for r in records]):.0f}")
    p(f"  Oscillation events       : {sum(1 for r in records if r.oscillation_detected)}")
    p(f"  No-turn events           : {sum(1 for r in records if r.no_turn_detected)}")
    p()

    # Section 2: Comparison with previous models
    p("=" * 60)
    p("2. COMPARISON WITH PREVIOUS MODELS")
    p("=" * 60)
    curr_succ  = len(successes) / total * 100
    curr_coll  = len(collisions) / total * 100
    curr_tout  = len(timeouts) / total * 100

    p(f"  {'Metric':<32} {'Original Champion':>18} {'Hybrid Phase4':>14} {'Phase5 Safety':>14}")
    p("  " + "-" * 82)
    p(f"  {'Overall Success (0.25m)':<32} {BASELINE_CHAMPION['total_success_pct']:>17.1f}% "
      f"{BASELINE_HYBRID['total_success_pct']:>13.1f}% {curr_succ:>13.1f}%")
    p(f"  {'Overall Collision':<32} {BASELINE_CHAMPION['total_collision_pct']:>17.1f}% "
      f"{BASELINE_HYBRID['total_collision_pct']:>13.1f}% {curr_coll:>13.1f}%")
    p(f"  {'Overall Timeout':<32} {BASELINE_CHAMPION['total_timeout_pct']:>17.1f}% "
      f"{BASELINE_HYBRID['total_timeout_pct']:>13.1f}% {curr_tout:>13.1f}%")
    # Category comparisons for open/wall/complex
    for cat_key, bl_key, cat_label in [
        ("EASY / OPEN", "open_success_pct", "Easy/Open Success"),
        ("WALL-BLOCKED", "wall_blocked_success_pct", "Wall-Blocked Success"),
        ("COMPLEX OBSTACLE", "complex_success_pct", "Complex Success"),
    ]:
        curr_cat_succ = cat_stats(cat_map.get(cat_key, [])).get("s_pct", 0.0)
        bl_champ = BASELINE_CHAMPION.get(bl_key)
        bl_hyb   = BASELINE_HYBRID.get(bl_key)
        p(f"  {cat_label:<32} "
          f"{(str(bl_champ)+'%') if bl_champ is not None else 'N/A':>18} "
          f"{'N/A':>14} "
          f"{curr_cat_succ:>13.1f}%")
    p()

    # Section 3: Failure count and percentage by cause
    p("=" * 60)
    p("3. FAILURE COUNT AND PERCENTAGE BY CAUSE")
    p("=" * 60)
    p(f"  Total failures : {len(failures)} / {total} ({len(failures)/total*100:.1f}%)")
    p()
    p(f"  {'Root Cause':<35} {'Count':>6} {'% of Failures':>14} {'% of Total':>11}")
    p("  " + "-" * 70)
    sorted_causes = sorted(failure_counts.items(), key=lambda x: -x[1])
    for cause, cnt in sorted_causes:
        p(f"  {cause:<35} {cnt:>6} {cnt/len(failures)*100:>13.1f}% {cnt/total*100:>10.1f}%")
    p()

    # Section 4: Evidence for each failure category
    p("=" * 60)
    p("4. EVIDENCE FOR EACH FAILURE CATEGORY")
    p("=" * 60)
    for cause, _ in sorted_causes:
        cause_records = [r for r in failures if r.failure_reason == cause]
        p(f"\n  [{cause}] — {len(cause_records)} episodes")
        for r in cause_records[:5]:   # Show up to 5 examples
            p(f"    • {r.scenario_id} | {r.category} | dist={r.final_dist_to_goal:.3f}m | "
              f"clear={r.min_clearance:.3f}m | lin={r.mean_linear_vel:.3f}m/s")
            p(f"      Evidence: {r.failure_evidence}")
        if len(cause_records) > 5:
            p(f"    ... and {len(cause_records)-5} more (see final_failure_diagnostics.csv)")
    p()

    # Section 5: Most common remaining failure
    p("=" * 60)
    p("5. MOST COMMON REMAINING FAILURE")
    p("=" * 60)
    if sorted_causes:
        top_cause, top_cnt = sorted_causes[0]
        p(f"  Primary failure  : {top_cause}")
        p(f"  Count            : {top_cnt} / {len(failures)} failures ({top_cnt/len(failures)*100:.1f}%)")
        if len(sorted_causes) > 1:
            sec_cause, sec_cnt = sorted_causes[1]
            p(f"  Secondary failure: {sec_cause}")
            p(f"  Count            : {sec_cnt} / {len(failures)} failures ({sec_cnt/len(failures)*100:.1f}%)")
    p()

    # Section 6: Nav2 vs PPO bottleneck
    p("=" * 60)
    p("6. BOTTLENECK ANALYSIS: NAV2 vs PPO")
    p("=" * 60)
    ppo_failures = [r for r in failures if r.failure_reason in {
        "LATE_OBSTACLE_REACTION", "EXCESSIVE_SPEED", "WRONG_LOCAL_MANEUVER",
        "OSCILLATION", "FAILED_PATH_FOLLOWING", "PATH_DEVIATION"
    }]
    nav2_failures = [r for r in failures if r.failure_reason in {
        "UNSAFE_NAV2_PATH", "TIMEOUT_LOW_PROGRESS"
    }]
    other_failures = [r for r in failures if r.failure_reason == "OTHER"]
    p(f"  PPO local-control failures  : {len(ppo_failures)} ({len(ppo_failures)/len(failures)*100:.1f}% of failures)")
    p(f"  Nav2 / path-level failures  : {len(nav2_failures)} ({len(nav2_failures)/len(failures)*100:.1f}% of failures)")
    p(f"  Unclassified (OTHER)        : {len(other_failures)} ({len(other_failures)/len(failures)*100:.1f}% of failures)")
    p()
    if len(ppo_failures) > len(nav2_failures):
        p("  CONCLUSION: PPO LOCAL CONTROL is the primary bottleneck.")
        p("  Nav2 path planning is not the limiting factor.")
    elif len(nav2_failures) > len(ppo_failures):
        p("  CONCLUSION: NAV2 PATH PLANNING is the primary bottleneck.")
    else:
        p("  CONCLUSION: Failures split evenly between PPO local control and Nav2.")
    p()

    # Section 7: Is further fine-tuning justified?
    p("=" * 60)
    p("7. IS FURTHER FINE-TUNING JUSTIFIED?")
    p("=" * 60)
    if curr_succ >= 85.0:
        p(f"  Success rate is {curr_succ:.1f}% >= 85%. Performance is EXCELLENT.")
        p("  Fine-tuning would have marginal gains. NOT strongly justified.")
        p("  Any further work should focus on architectural improvements (e.g., LSTM memory).")
    elif curr_succ >= 70.0:
        p(f"  Success rate is {curr_succ:.1f}%. Performance is GOOD but improvement room exists.")
        p("  Further fine-tuning IS justified if the top failure cause is addressable.")
        if sorted_causes:
            p(f"  Top cause '{sorted_causes[0][0]}' may be addressable with targeted reward shaping.")
    else:
        p(f"  Success rate is {curr_succ:.1f}%. Significant room for improvement.")
        p("  Further fine-tuning IS justified.")
        if sorted_causes:
            p(f"  Priority target: '{sorted_causes[0][0]}'.")
    p()

    # Section 8: Final recommendation
    p("=" * 60)
    p("8. FINAL RECOMMENDATION")
    p("=" * 60)
    if curr_succ >= 90.0:
        verdict = "COMPLETE"
        p(f"  Verdict: {verdict}")
        p(f"  The Phase 5 Safety-Margin model achieves {curr_succ:.1f}% success on 150 unseen")
        p(f"  deterministic scenarios at the strict 0.25 m threshold.")
        p("  This represents a strong, deployment-ready agent. No further fine-tuning required.")
    elif curr_succ >= 70.0:
        verdict = "CONTINUE (optional)"
        p(f"  Verdict: {verdict}")
        p(f"  The model achieves {curr_succ:.1f}% success. Strong but not exceptional.")
        p(f"  Consider one more fine-tuning round targeting '{sorted_causes[0][0] if sorted_causes else 'top cause'}'.")
    else:
        verdict = "CONTINUE (recommended)"
        p(f"  Verdict: {verdict}")
        p(f"  The model achieves {curr_succ:.1f}% success. Further training recommended.")

    p()
    p("=" * 60)
    p("END OF REPORT")
    p("=" * 60)

    report_text = "\n".join(lines)
    with open(OUT_REPORT_TXT, "w") as f:
        f.write(report_text)
    print(f"  [SAVED] {OUT_REPORT_TXT}")

    # Print full report to console
    print("\n" + report_text)

    print("\n[EVALUATION COMPLETE] All 3 output files saved to diagnostics/")
    return verdict


if __name__ == "__main__":
    run_evaluation()
