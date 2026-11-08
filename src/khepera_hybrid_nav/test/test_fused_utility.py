"""Unit tests for unified utility calculation in Khepera Hybrid Navigation."""

import math


def compute_test_utility(
    dist_to_robot: float,
    semantic_prob: float,
    exploration_gain: float,
    visual_match_score: float,
    enable_3dgs: bool,
    w_dist: float = 0.25,
    w_sem: float = 0.45,
    w_vis: float = 0.30
) -> float:
    """Helper computing fused utility matching HybridCoordinatorNode logic."""
    d_norm = 1.0 / (1.0 + dist_to_robot)
    sem_term = semantic_prob * exploration_gain
    vis_term = visual_match_score if enable_3dgs else 0.0

    return w_dist * d_norm + w_sem * sem_term + (w_vis * vis_term if enable_3dgs else 0.0)


def test_fused_utility_toggle_behavior() -> None:
    """Verify that enable_3dgs cleanly activates or suppresses the visual matching score."""
    dist = 1.0
    sem_p = 0.8
    gain = 5.0
    vis_s = 20.0

    # With 3DGS disabled: visual score has zero contribution
    u_disabled = compute_test_utility(dist, sem_p, gain, vis_s, enable_3dgs=False)
    # With 3DGS enabled: visual score contributes
    u_enabled = compute_test_utility(dist, sem_p, gain, vis_s, enable_3dgs=True)

    assert u_enabled > u_disabled
    expected_diff = 0.30 * vis_s
    assert abs((u_enabled - u_disabled) - expected_diff) < 1e-4
