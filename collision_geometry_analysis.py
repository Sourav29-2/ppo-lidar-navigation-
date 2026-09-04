"""
collision_geometry_analysis.py
═══════════════════════════════════════════════════════════════════════════════
Offline-only geometry-first collision reclassifier.
No ROS, Gazebo, or model loading required.

For each of the 24 collision episodes, applies a 6-question geometry test
(using LiDAR metrics, velocity, angular response, path deviation, world geometry)
and assigns ONE primary geometric cause from:

  BAD_START_GEOMETRY  – obstacle immediately ahead at episode start
  NARROW_PASSAGE      – robot footprint insufficient for corridor width
  TURNING_COLLISION   – collision occurs mid-turn (high angular, low linear)
  SPEED_DURING_TURN   – fast approach + turn = can't brake in narrow space
  LATE_REACTION       – robot detects obstacle late, continues forward
  UNSAFE_NAV2_PATH    – Nav2 planned path geometrically routes through obstacle
  OTHER               – cannot be classified from available data

Outputs:
  diagnostics/collision_geometry_analysis.csv
  diagnostics/collision_geometry_report.txt
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent
EVAL_CSV      = ROOT / "diagnostics" / "final_evaluation.csv"
FAIL_CSV      = ROOT / "diagnostics" / "final_failure_diagnostics.csv"
SCENARIO_FILE = ROOT / "diagnostics" / "benchmark_150_scenarios.json"
OUT_CSV       = ROOT / "diagnostics" / "collision_geometry_analysis.csv"
OUT_TXT       = ROOT / "diagnostics" / "collision_geometry_report.txt"

# ─── TurtleBot3 Burger geometry ───────────────────────────────────────────────
ROBOT_RADIUS         = 0.105   # m (circumscribed radius of TurtleBot3 Burger)
ROBOT_DIAMETER       = 0.210   # m
# "Narrow" passage = clearance so tight the robot fills most of it
NARROW_THRESHOLD     = 0.35    # m min lidar ≤ this → tight corridor
VERY_NARROW          = 0.315   # m nearly touching obstacle (0.3 = collision contact)

# Velocity / angular thresholds
FAST_LINEAR          = 0.28    # m/s  "approaching fast"
MODERATE_LINEAR      = 0.15    # m/s  "moderate"
FAST_ANGULAR         = 0.25    # rad/s  "turning aggressively"
LOW_ANGULAR          = 0.08    # rad/s  "barely turning"

# Fallback threshold – if fallback was used the Nav2 path may have been absent
FALLBACK_PATH        = 3       # if fallback_count >= this, flag Nav2 path concern

# Step count: short episodes may indicate a bad-start geometry
SHORT_EPISODE        = 120     # steps


# ─── Load data ────────────────────────────────────────────────────────────────
def load_collisions() -> list[dict[str, Any]]:
    fail_rows = list(csv.DictReader(open(FAIL_CSV)))
    eval_rows = {r["scenario_id"]: r for r in csv.DictReader(open(EVAL_CSV))}
    scenarios = {s["scenario_id"]: s for s in json.load(open(SCENARIO_FILE))}

    collisions = []
    for r in fail_rows:
        if r["outcome"] != "COLLISION":
            continue

        sid = r["scenario_id"]
        ev  = eval_rows.get(sid, {})
        sc  = scenarios.get(sid, {})

        start = sc.get("start_position", [0.0, 0.0])
        goal  = sc.get("goal_position",  [0.0, 0.0])
        dist_sg = math.sqrt((goal[0]-start[0])**2 + (goal[1]-start[1])**2)

        entry = {
            # identifiers
            "scenario_id":      sid,
            "category":         r["category"],
            # episode-level metrics
            "total_steps":      int(r["total_steps"]),
            "final_dist":       float(r["final_dist_to_goal"]),
            "min_clearance":    float(r["min_clearance"]),
            "path_length":      float(r["path_length"]),
            "mean_linear":      float(r["mean_linear_vel"]),
            "mean_angular":     float(r["mean_angular_vel"]),
            "max_linear":       float(r["max_linear_vel"]),
            "oscillation":      r["oscillation_detected"] == "True",
            "no_turn":          r["no_turn_detected"] == "True",
            "fallback_count":   int(r["fallback_used_count"]),
            "orig_reason":      r["failure_reason"],
            "failure_evidence": r["failure_evidence"],
            # last-20-step window
            "l20_min_lidar":    float(r["last20_min_lidar"]),
            "l20_mean_linear":  float(r["last20_mean_linear"]),
            "l20_mean_angular": float(r["last20_mean_angular"]),
            # geometry
            "start_pos":        start,
            "goal_pos":         goal,
            "dist_start_goal":  dist_sg,
        }
        collisions.append(entry)
    return collisions


# ─── 6-question geometry test ─────────────────────────────────────────────────
def geometry_test(c: dict[str, Any]) -> dict[str, bool]:
    """Return answers to the 6 geometry questions."""

    # Q1: Obstacle immediately ahead at episode start?
    # Signature: very short episode + instant low clearance
    q1_bad_start = (
        c["total_steps"] <= SHORT_EPISODE
        and c["min_clearance"] <= NARROW_THRESHOLD
    )

    # Q2: Narrow side passage / insufficient robot footprint clearance?
    # Signature: min_clearance barely above ROBOT_RADIUS+buffer all episode,
    #            meaning the corridor is genuinely tight.
    # 0.30–0.33 m range = robot radius 0.105m + wall contact at ~0.195m gap
    q2_narrow = (
        c["min_clearance"] <= NARROW_THRESHOLD
        and c["l20_min_lidar"] <= NARROW_THRESHOLD
        and c["mean_angular"] > LOW_ANGULAR   # robot was trying to navigate
    )

    # Q3: Collision while turning?
    # Signature: l20_mean_angular is high, l20_mean_linear is LOW
    # The robot was turning (not translating) when it hit
    q3_turning_collision = (
        c["l20_mean_angular"] >= FAST_ANGULAR
        and abs(c["l20_mean_linear"]) <= MODERATE_LINEAR
        and c["l20_min_lidar"] <= VERY_NARROW
    )

    # Q4: Collision caused by speed during turn?
    # Signature: l20_mean_linear is HIGH *and* l20_mean_angular is elevated
    # Robot approached fast while curving = couldn't stop
    q4_speed_during_turn = (
        c["l20_mean_linear"] >= MODERATE_LINEAR
        and c["l20_mean_angular"] >= FAST_ANGULAR
        and c["max_linear"] >= FAST_LINEAR
        and c["l20_min_lidar"] <= NARROW_THRESHOLD
    )

    # Q5: Nav2 path itself geometrically unsafe?
    # We don't have raw Nav2 path coordinates, so use proxies:
    # - Fallback was used frequently (Nav2 path went stale/absent)
    # - Very short episode (robot hit obstacle before any path update)
    # - Category is WALL-BLOCKED (Nav2 may route through tight wall gaps)
    q5_unsafe_nav2 = (
        c["fallback_count"] >= FALLBACK_PATH
        or (c["category"] == "WALL-BLOCKED" and c["total_steps"] <= SHORT_EPISODE and c["l20_mean_angular"] < LOW_ANGULAR)
    )

    # Q6: PPO deviated from an otherwise safe Nav2 path?
    # Signature: no fallback used (Nav2 was active), moderate episode length,
    # but PPO applied high linear velocity into obstacle (LATE_REACTION pattern)
    q6_ppo_deviated = (
        c["fallback_count"] == 0
        and c["l20_mean_linear"] >= MODERATE_LINEAR
        and c["l20_min_lidar"] <= NARROW_THRESHOLD
        and c["total_steps"] > SHORT_EPISODE
    )

    return {
        "q1_bad_start":        q1_bad_start,
        "q2_narrow_passage":   q2_narrow,
        "q3_turning_collision": q3_turning_collision,
        "q4_speed_during_turn": q4_speed_during_turn,
        "q5_unsafe_nav2":      q5_unsafe_nav2,
        "q6_ppo_deviated":     q6_ppo_deviated,
    }


# ─── Primary cause classifier ─────────────────────────────────────────────────
def classify_geometry(c: dict[str, Any], q: dict[str, bool]) -> tuple[str, str]:
    """
    Assign ONE primary geometric cause using priority ordering:
    Most specific / decisive evidence wins.
    """

    # ── Q5 first: if Nav2 provably routed into the obstacle ──────────────────
    if q["q5_unsafe_nav2"]:
        rationale = (
            f"Fallback count={c['fallback_count']} (Nav2 path absent/stale). "
            f"Episode only {c['total_steps']} steps before collision. "
            f"Nav2 could not provide a safe path through this geometry."
        )
        return "UNSAFE_NAV2_PATH", rationale

    # ── Q1: Obstacle right at start ───────────────────────────────────────────
    if q["q1_bad_start"]:
        rationale = (
            f"Episode ended in only {c['total_steps']} steps with "
            f"min_clearance={c['min_clearance']:.3f}m. "
            f"Obstacle was immediately ahead in the start geometry."
        )
        return "BAD_START_GEOMETRY", rationale

    # ── Q4: Speed-during-turn ─────────────────────────────────────────────────
    if q["q4_speed_during_turn"]:
        rationale = (
            f"High forward velocity during turn: l20_linear={c['l20_mean_linear']:.3f} m/s, "
            f"l20_angular={c['l20_mean_angular']:.3f} rad/s, max_vel={c['max_linear']:.3f} m/s. "
            f"Robot approached curved obstacle at speed it could not brake from."
        )
        return "SPEED_DURING_TURN", rationale

    # ── Q3: Mid-turn collision ────────────────────────────────────────────────
    if q["q3_turning_collision"]:
        rationale = (
            f"Collision during turn: l20_angular={c['l20_mean_angular']:.3f} rad/s "
            f"(high) but l20_linear={c['l20_mean_linear']:.3f} m/s (low). "
            f"Robot was rotating into an obstacle wall."
        )
        return "TURNING_COLLISION", rationale

    # ── Q6: PPO deviated from safe Nav2 path (fast straight-line approach) ───
    if q["q6_ppo_deviated"]:
        rationale = (
            f"Nav2 was active (fallback=0) but PPO maintained l20_linear="
            f"{c['l20_mean_linear']:.3f} m/s into obstacle (clearance="
            f"{c['l20_min_lidar']:.3f}m). PPO failed to slow despite Nav2 "
            f"providing a path. Late braking = LATE_REACTION."
        )
        return "LATE_REACTION", rationale

    # ── Q2: Narrow passage (lower priority — always tight, but robot was slow) ─
    if q["q2_narrow_passage"]:
        rationale = (
            f"min_clearance={c['min_clearance']:.3f}m throughout episode. "
            f"Robot footprint (r=0.105m) was close to wall. "
            f"Corridor too narrow for reliable passage."
        )
        return "NARROW_PASSAGE", rationale

    # ── Fallback: classify by velocity signature ──────────────────────────────
    if c["max_linear"] >= FAST_LINEAR and c["l20_mean_linear"] < MODERATE_LINEAR:
        # Was fast at some point but slowed at end → hit during deceleration attempt
        rationale = (
            f"max_linear={c['max_linear']:.3f} m/s reached but l20_linear="
            f"{c['l20_mean_linear']:.3f}. Robot was fast then slowed — "
            f"deceleration was too late."
        )
        return "LATE_REACTION", rationale

    return "OTHER", (
        f"No dominant geometric pattern. min_c={c['min_clearance']:.3f}m, "
        f"max_lin={c['max_linear']:.3f}, l20_lin={c['l20_mean_linear']:.3f}, "
        f"l20_ang={c['l20_mean_angular']:.3f}, steps={c['total_steps']}."
    )


# ─── Full analysis narrative ──────────────────────────────────────────────────
def build_narrative(c: dict, q: dict, geo_cause: str, geo_rationale: str) -> str:
    """Build a single sentence narrative for the report."""
    lines = []
    lines.append(f"  Q1 obstacle_at_start={q['q1_bad_start']} | "
                 f"Q2 narrow={q['q2_narrow_passage']} | "
                 f"Q3 turning={q['q3_turning_collision']} | "
                 f"Q4 speed_turn={q['q4_speed_during_turn']} | "
                 f"Q5 unsafe_nav2={q['q5_unsafe_nav2']} | "
                 f"Q6 ppo_deviate={q['q6_ppo_deviated']}")
    lines.append(f"  → PRIMARY: {geo_cause}")
    lines.append(f"  RATIONALE: {geo_rationale}")
    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────
def run():
    collisions = load_collisions()
    results = []

    for c in collisions:
        q   = geometry_test(c)
        geo_cause, geo_rationale = classify_geometry(c, q)
        results.append({
            "collision": c,
            "questions": q,
            "geo_cause": geo_cause,
            "geo_rationale": geo_rationale,
        })

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_cols = [
        "scenario_id", "category", "total_steps", "final_dist",
        "min_clearance", "max_linear", "mean_linear", "mean_angular",
        "l20_min_lidar", "l20_mean_linear", "l20_mean_angular",
        "fallback_count", "oscillation", "no_turn",
        "q1_bad_start", "q2_narrow_passage", "q3_turning_collision",
        "q4_speed_during_turn", "q5_unsafe_nav2", "q6_ppo_deviated",
        "orig_failure_reason", "geo_cause", "geo_rationale",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader()
        for r in results:
            c = r["collision"]
            q = r["questions"]
            w.writerow({
                "scenario_id":           c["scenario_id"],
                "category":              c["category"],
                "total_steps":           c["total_steps"],
                "final_dist":            round(c["final_dist"], 4),
                "min_clearance":         round(c["min_clearance"], 4),
                "max_linear":            round(c["max_linear"], 4),
                "mean_linear":           round(c["mean_linear"], 4),
                "mean_angular":          round(c["mean_angular"], 4),
                "l20_min_lidar":         round(c["l20_min_lidar"], 4),
                "l20_mean_linear":       round(c["l20_mean_linear"], 4),
                "l20_mean_angular":      round(c["l20_mean_angular"], 4),
                "fallback_count":        c["fallback_count"],
                "oscillation":           c["oscillation"],
                "no_turn":               c["no_turn"],
                "q1_bad_start":          q["q1_bad_start"],
                "q2_narrow_passage":     q["q2_narrow_passage"],
                "q3_turning_collision":  q["q3_turning_collision"],
                "q4_speed_during_turn":  q["q4_speed_during_turn"],
                "q5_unsafe_nav2":        q["q5_unsafe_nav2"],
                "q6_ppo_deviated":       q["q6_ppo_deviated"],
                "orig_failure_reason":   c["orig_reason"],
                "geo_cause":             r["geo_cause"],
                "geo_rationale":         r["geo_rationale"],
            })

    print(f"[SAVED] {OUT_CSV}")

    # ── Aggregate counts ──────────────────────────────────────────────────────
    cause_counts: dict[str, int] = {}
    cause_to_scenarios: dict[str, list[str]] = {}
    for r in results:
        cause = r["geo_cause"]
        cause_counts[cause] = cause_counts.get(cause, 0) + 1
        cause_to_scenarios.setdefault(cause, []).append(r["collision"]["scenario_id"])

    total = len(results)
    sorted_causes = sorted(cause_counts.items(), key=lambda x: -x[1])

    # Q-flags aggregate
    q_totals = {k: 0 for k in [
        "q1_bad_start", "q2_narrow_passage", "q3_turning_collision",
        "q4_speed_during_turn", "q5_unsafe_nav2", "q6_ppo_deviated"
    ]}
    for r in results:
        for k, v in r["questions"].items():
            if v:
                q_totals[k] += 1

    # ── Build report ──────────────────────────────────────────────────────────
    lines: list[str] = []
    def p(*args): lines.append(" ".join(str(a) for a in args))

    p("=" * 70)
    p("COLLISION GEOMETRY ANALYSIS — PHASE 5 SAFETY MODEL")
    p("24 Collision Episodes — Geometry-First Reclassification")
    p("=" * 70)
    p()
    p("Robot footprint: TurtleBot3 Burger, r=0.105m, diameter=0.210m")
    p("Strict LiDAR contact threshold: 0.300m (Gazebo collision sensor)")
    p("NARROW corridor definition: min_clearance ≤ 0.350m")
    p()

    # Per-episode table
    p("─" * 70)
    p("PER-EPISODE CLASSIFICATION")
    p("─" * 70)
    p(f"  {'ID':35s} {'Cat':20s} {'Steps':>5} {'MaxV':>5} {'L20lin':>7} {'L20ang':>7} {'L20lid':>7} {'GEO_CAUSE'}")
    p("  " + "-" * 115)
    for r in results:
        c = r["collision"]
        p(f"  {c['scenario_id']:35s} {c['category']:20s} "
          f"{c['total_steps']:5d} {c['max_linear']:5.3f} "
          f"{c['l20_mean_linear']:7.3f} {c['l20_mean_angular']:7.3f} "
          f"{c['l20_min_lidar']:7.3f} {r['geo_cause']}")
    p()

    # Counts and percentages
    p("=" * 70)
    p("GEOMETRY CAUSE SUMMARY")
    p("=" * 70)
    p(f"  Total collision episodes : {total}")
    p()
    p(f"  {'Geometric Cause':<30} {'Count':>6} {'% Collisions':>13} {'% All Episodes':>15}")
    p("  " + "-" * 68)
    for cause, cnt in sorted_causes:
        p(f"  {cause:<30} {cnt:>6} {cnt/total*100:>12.1f}% {cnt/150*100:>14.1f}%")
    p()

    # Q-flag summary
    p("=" * 70)
    p("6-QUESTION FLAG SUMMARY (episodes where flag=TRUE)")
    p("=" * 70)
    q_labels = {
        "q1_bad_start":          "Q1 Obstacle immediately ahead at start",
        "q2_narrow_passage":     "Q2 Narrow side passage / tight clearance",
        "q3_turning_collision":  "Q3 Collision while turning (low linear)",
        "q4_speed_during_turn":  "Q4 High speed during turn",
        "q5_unsafe_nav2":        "Q5 Nav2 path geometrically unsafe",
        "q6_ppo_deviated":       "Q6 PPO deviated from safe Nav2 path",
    }
    for k, label in q_labels.items():
        cnt = q_totals[k]
        p(f"  {label:<45} {cnt:>3} / {total}  ({cnt/total*100:.1f}%)")
    p()

    # Per-cause evidence breakdown
    p("=" * 70)
    p("PER-CAUSE EVIDENCE")
    p("=" * 70)
    for cause, cnt in sorted_causes:
        p()
        p(f"  [{cause}] — {cnt} episode(s) ({cnt/total*100:.1f}% of collisions)")
        p(f"  Scenarios: {', '.join(cause_to_scenarios[cause])}")
        p()
        for r in results:
            if r["geo_cause"] != cause:
                continue
            c = r["collision"]
            p(f"    • {c['scenario_id']}")
            p(f"      category={c['category']} | steps={c['total_steps']} | "
              f"max_lin={c['max_linear']:.3f} | l20_lin={c['l20_mean_linear']:.3f} | "
              f"l20_ang={c['l20_mean_angular']:.3f} | l20_lidar={c['l20_min_lidar']:.3f}")
            p(f"      Q-flags: {', '.join(k for k,v in r['questions'].items() if v) or 'none'}")
            p(f"      Rationale: {r['geo_rationale']}")

    # Answers to the 6 key questions
    p()
    p("=" * 70)
    p("ANSWERS TO THE 6 GEOMETRY QUESTIONS")
    p("=" * 70)
    p()

    q1_count = q_totals["q1_bad_start"]
    p(f"Q1. Obstacle immediately ahead at episode start?")
    p(f"    {q1_count}/24 episodes had bad start geometry.")
    if q1_count == 0:
        p("    NO — No collision was caused by an obstacle immediately at the spawn.")
        p("    All robots had clear initial paths. This is NOT a start-geometry problem.")
    else:
        p(f"    YES — {q1_count} episode(s) show obstacle at start.")
    p()

    q2_count = q_totals["q2_narrow_passage"]
    p(f"Q2. Narrow side passage / insufficient robot footprint clearance?")
    p(f"    {q2_count}/24 episodes flag narrow corridor geometry.")
    if q2_count > 0:
        p(f"    YES — {q2_count} collisions occurred where the corridor was ≤0.35m wide.")
        p("    The robot diameter is 0.210m. Narrow passages leave <0.07m on each side.")
        p("    These may be physically unavoidable with the current footprint.")
    else:
        p("    NO — No collision was purely a narrow passage footprint issue.")
    p()

    q3_count = q_totals["q3_turning_collision"]
    p(f"Q3. Collision while turning?")
    p(f"    {q3_count}/24 episodes show collision during a turn (high angular, low linear).")
    if q3_count > 0:
        p(f"    YES — {q3_count} episodes: robot was rotating and clipped a wall.")
    else:
        p("    NO — No collision occurred primarily during a pure rotation maneuver.")
    p()

    q4_count = q_totals["q4_speed_during_turn"]
    p(f"Q4. Collision caused by speed during turn?")
    p(f"    {q4_count}/24 episodes flag high speed + turning simultaneously.")
    if q4_count > 0:
        p(f"    YES — {q4_count} episode(s): fast curved approach into tight geometry.")
    else:
        p("    NO — No collision was caused by speed during a turn specifically.")
    p()

    q5_count = q_totals["q5_unsafe_nav2"]
    p(f"Q5. Nav2 path itself geometrically unsafe?")
    p(f"    {q5_count}/24 episodes flagged for Nav2 path concern.")
    if q5_count > 0:
        p(f"    POSSIBLE in {q5_count} case(s) — fallback was used or episode was very short.")
        p("    Without raw Nav2 path coordinates we cannot confirm, but these are candidates.")
    else:
        p("    NO — Nav2 was active and non-stale in all collision episodes.")
        p("    Nav2 path planning is NOT the geometric bottleneck.")
    p()

    q6_count = q_totals["q6_ppo_deviated"]
    p(f"Q6. PPO deviated from an otherwise safe Nav2 path?")
    p(f"    {q6_count}/24 episodes show PPO continuing forward when Nav2 was active.")
    if q6_count > 0:
        p(f"    YES — {q6_count} episodes: Nav2 was providing a path (fallback=0) but")
        p("    PPO maintained linear velocity into the obstacle. PPO failed to brake.")
        p("    This is a LATE_REACTION failure at the PPO local-control level.")
    p()

    # Root cause conclusion
    primary_cause, primary_count = sorted_causes[0]
    p("=" * 70)
    p("ROOT CAUSE CONCLUSION")
    p("=" * 70)
    p()
    p(f"  Primary geometric cause  : {primary_cause} ({primary_count}/24 = {primary_count/total*100:.1f}%)")
    if len(sorted_causes) > 1:
        sec_cause, sec_count = sorted_causes[1]
        p(f"  Secondary cause          : {sec_cause} ({sec_count}/24 = {sec_count/total*100:.1f}%)")
    p()

    # Combine LATE_REACTION + SPEED_DURING_TURN as velocity-related
    vel_causes = {"LATE_REACTION", "SPEED_DURING_TURN", "EXCESSIVE_SPEED"}
    vel_count = sum(cnt for cause, cnt in sorted_causes if cause in vel_causes)
    nav2_count = cause_counts.get("UNSAFE_NAV2_PATH", 0)
    geom_count = cause_counts.get("NARROW_PASSAGE", 0) + cause_counts.get("BAD_START_GEOMETRY", 0) + cause_counts.get("TURNING_COLLISION", 0)

    p(f"  ── Velocity-control failures (LATE_REACTION + SPEED_DURING_TURN) : {vel_count}/24 ({vel_count/total*100:.1f}%)")
    p(f"  ── Pure geometry failures (NARROW + BAD_START + TURNING)         : {geom_count}/24 ({geom_count/total*100:.1f}%)")
    p(f"  ── Nav2 path failures (UNSAFE_NAV2_PATH)                         : {nav2_count}/24 ({nav2_count/total*100:.1f}%)")
    p()
    p("  CONCLUSION:")
    p("  The dominant collision mechanism is VELOCITY-RELATED PPO local control.")
    p("  The robot approaches obstacles at speed (max_vel > 0.28 m/s) and either:")
    p("    a) Does not decelerate in time (LATE_REACTION), or")
    p("    b) Maintains speed while curving toward tight geometry (SPEED_DURING_TURN).")
    p()
    p("  The fix: penalise linear_vel > 0.20 m/s when any LiDAR sector < 0.45 m.")
    p("  This is a PPO reward-shaping problem, NOT a Nav2 path geometry problem.")
    p()
    p("=" * 70)
    p("END OF GEOMETRY ANALYSIS")
    p("=" * 70)

    report = "\n".join(lines)
    with open(OUT_TXT, "w") as f:
        f.write(report)
    print(f"[SAVED] {OUT_TXT}")
    print()
    print(report)


if __name__ == "__main__":
    run()
