"""Unit tests for OpenFrontier GlobalFrontierManager."""

from openfrontier_ros.frontier_manager import GlobalFrontierManager, ManagedFrontier


def test_frontier_merging_and_pruning() -> None:
    """Verify spatial merging of nearby proposals and pruning near robot."""
    mgr = GlobalFrontierManager(tau_merge_m=1.8, r_clear_m=0.5)

    f1 = ManagedFrontier(
        id="A",
        pos_x=2.0,
        pos_y=2.0,
        pos_z=0.0,
        pixel_u=100,
        pixel_v=100,
        exploration_gain=5.0
    )
    # Proposal close to f1 (< 1.8 m distance)
    f2 = ManagedFrontier(
        id="B",
        pos_x=2.5,
        pos_y=2.2,
        pos_z=0.0,
        pixel_u=120,
        pixel_v=110,
        exploration_gain=8.0
    )
    # Proposal far from f1 (> 1.8 m distance)
    f3 = ManagedFrontier(
        id="C",
        pos_x=5.0,
        pos_y=5.0,
        pos_z=0.0,
        pixel_u=300,
        pixel_v=200,
        exploration_gain=3.0
    )

    mgr.merge_proposals([f1, f2, f3])
    active = mgr.get_active_frontiers()

    # f1 and f2 should be merged into 1, f3 separate -> 2 active frontiers
    assert len(active) == 2

    # Prune frontiers near robot at (2.1, 2.1) within r_clear 0.5m
    mgr.prune_cleared_frontiers(robot_x=2.1, robot_y=2.1)
    remaining = mgr.get_active_frontiers()
    assert len(remaining) == 1
    assert remaining[0].id == "C"


def test_stall_detection() -> None:
    """Verify that HandleProgress detects lack of distance progress."""
    mgr = GlobalFrontierManager()

    # Repeatedly supply same distance without progress
    stalled = False
    for _ in range(16):
        stalled = mgr.handle_progress(current_distance=2.0)
        if stalled:
            break

    assert stalled is True
