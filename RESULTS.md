# Evaluation Results

## Final Model — Phase 5 Safety Champion

**82.7% success rate** over 150 diverse navigation scenarios in a simulated
16 × 16 m indoor apartment with walls, furniture, and tight corridors.

- **Strict success criterion**: distance to goal ≤ 0.25 m
- **Zero oscillations** detected across all 150 runs
- **Zero fallback triggers** — policy is fully self-sufficient
- **Mean clearance**: 0.44 m (safely away from walls)

---

## Model Progression

| Model | Overall Success | Collision | Timeout | Notes |
|---|---|---|---|---|
| Original PPO-only | 52.0% | 32.7% | 15.3% | 40D obs, no Nav2 |
| Hybrid Phase 4 | 64.2% | 26.3% | 9.5% | 50D obs, Nav2 path added |
| **Phase 5 Safety** | **82.7%** | **16.0%** | **1.3%** | Safety margin fine-tune |

---

## Per-Category Breakdown (Phase 5)

| Scenario Category | N | Success | Collision | Timeout |
|---|---|---|---|---|
| 🏆 TOP_LEFT | 10 | **100.0%** | 0.0% | 0.0% |
| EASY / OPEN | 30 | 90.0% | 10.0% | 0.0% |
| RANDOM | 20 | 90.0% | 10.0% | 0.0% |
| BOTTOM_LEFT | 10 | 90.0% | 10.0% | 0.0% |
| BOTTOM_RIGHT | 10 | 90.0% | 10.0% | 0.0% |
| WALL-BLOCKED | 30 | 73.3% | 23.3% | 3.3% |
| COMPLEX OBSTACLE | 30 | 73.3% | 26.7% | 0.0% |
| TOP_RIGHT | 10 | 70.0% | 20.0% | 10.0% |

---

## Failure Root-Cause Analysis

| Root Cause | Count | % of Failures |
|---|---|---|
| EXCESSIVE_SPEED | 14 | 53.8% |
| LATE_OBSTACLE_REACTION | 7 | 26.9% |
| WRONG_LOCAL_MANEUVER | 3 | 11.5% |
| TIMEOUT_LOW_PROGRESS | 2 | 7.7% |

**Key insight**: 92.3% of all failures are PPO local-control issues (speed management near
obstacles). Nav2 path planning is not the bottleneck — the global plans are consistently good.

---

## Benchmark Scenarios

The 150 evaluation scenarios are defined in [`results/benchmark_150_scenarios.json`](results/benchmark_150_scenarios.json)
and cover 8 categories designed to test different navigation challenges:

- **EASY / OPEN**: Clear paths to goals in open spaces
- **WALL-BLOCKED**: Goals accessible only through narrow wall gaps
- **COMPLEX OBSTACLE**: Dense furniture clusters between robot and goal
- **TOP_LEFT / TOP_RIGHT / BOTTOM_LEFT / BOTTOM_RIGHT**: Corner-to-corner traversals
- **RANDOM**: Random start/goal pairs anywhere in the map

---

## Raw Results

- [`results/final_evaluation.csv`](results/final_evaluation.csv) — per-episode metrics
- [`results/final_failure_diagnostics.csv`](results/final_failure_diagnostics.csv) — failure evidence per episode
- [`results/final_evaluation_report.txt`](results/final_evaluation_report.txt) — full text report
