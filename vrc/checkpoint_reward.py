"""
Checkpoint Reward Computation for Verifiable Reward Chains (VRC).

Core module: computes dense intermediate reward by matching agent trajectories
against ordered state checkpoint predicates.

Two main matching modes:
  - gated:     credit for P_i only if ALL preceding P_j matched (default, anti-reward-hacking)
  - unordered: simple hit count, no ordering constraint (ablation baseline)
"""

from __future__ import annotations

from typing import Callable, List, Optional


StatePredicate = Callable[[str], bool]


def compute_checkpoint_reward(
    observations: List[str],
    predicates: List[StatePredicate],
    mode: str = "gated",
    weights: Optional[List[float]] = None,
) -> float:
    """
    Compute reward in [0, 1] for a trajectory's observations against ordered checkpoint predicates.

    Args:
        observations: list of observation strings from a single trajectory
        predicates: ordered list of state predicate functions [P_1, ..., P_n]
        mode: "gated" | "unordered"
        weights: per-checkpoint weights (default: uniform, sum to 1)

    Returns:
        Reward score in [0, 1]
    """
    n = len(predicates)
    if n == 0:
        return 0.0

    if weights is None:
        weights = [1.0 / n] * n

    # Check which predicates match ANY observation in the trajectory
    matches = []
    for predicate in predicates:
        matched = any(predicate(obs) for obs in observations)
        matches.append(matched)

    if mode == "gated":
        r = 0.0
        gate = 1.0
        for i in range(n):
            m = 1.0 if matches[i] else 0.0
            r += weights[i] * m * gate
            gate *= m
        return r
    elif mode == "unordered":
        r = 0.0
        for i in range(n):
            if matches[i]:
                r += weights[i]
        return r
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from: gated, unordered")
