"""Global Target Manager implementing OpenFrontier Algorithm 1."""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class ManagedFrontier:
    """Represents a tracked 3D frontier candidate in the global frame."""

    id: str
    pos_x: float
    pos_y: float
    pos_z: float
    pixel_u: int
    pixel_v: int
    exploration_gain: float
    semantic_prob: float = 0.5
    visual_match_score: float = 0.0
    total_utility: float = 0.0
    is_viewpoint_frontier: bool = False


@dataclass
class ProgressTracker:
    """Tracks robot progress towards target for stall detection."""

    stall_counter: int = 0
    d_prev: float = 1e9
    stall_cycles_threshold: int = 15
    epsilon_progress_m: float = 0.1


class GlobalFrontierManager:
    """Maintains active frontiers, handles spatial merging, pruning, and utility scoring."""

    def __init__(
        self,
        tau_merge_m: float = 1.8,
        r_clear_m: float = 0.5,
        r_near_m: float = 0.5,
        r_goal_m: float = 0.8
    ) -> None:
        """Initialize thresholds and tracking structures."""
        self._tau_merge = tau_merge_m
        self._r_clear = r_clear_m
        self._r_near = r_near_m
        self._r_goal = r_goal_m

        self._active_frontiers: Dict[str, ManagedFrontier] = {}
        self._cleared_positions: List[Tuple[float, float, float]] = []
        self._progress = ProgressTracker()
        self._next_id_idx = 0

    def get_active_frontiers(self) -> List[ManagedFrontier]:
        """Return list of all currently tracked active frontiers."""
        return list(self._active_frontiers.values())

    def generate_next_id(self) -> str:
        """Generate alphabetical or indexed identifier (A, B, ..., Z, AA, ...)."""
        idx = self._next_id_idx
        self._next_id_idx += 1
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if idx < len(chars):
            return chars[idx]
        return f"F{idx}"

    def merge_proposals(
        self,
        proposals: List[ManagedFrontier]
    ) -> None:
        """Merge newly detected frontiers into active set if distance >= tau_merge."""
        for prop in proposals:
            if self._is_near_cleared(prop.pos_x, prop.pos_y, prop.pos_z):
                continue

            matched_id = None
            for fid, existing in self._active_frontiers.items():
                dist = math.dist(
                    (prop.pos_x, prop.pos_y, prop.pos_z),
                    (existing.pos_x, existing.pos_y, existing.pos_z)
                )
                if dist < self._tau_merge:
                    matched_id = fid
                    break

            if matched_id is not None:
                existing = self._active_frontiers[matched_id]
                existing.exploration_gain = max(existing.exploration_gain, prop.exploration_gain)
                existing.pixel_u = prop.pixel_u
                existing.pixel_v = prop.pixel_v
            else:
                self._active_frontiers[prop.id] = prop

    def prune_cleared_frontiers(
        self,
        robot_x: float,
        robot_y: float
    ) -> None:
        """Remove active frontiers that are within r_clear of current robot pose."""
        to_remove: List[str] = []
        for fid, f in self._active_frontiers.items():
            dist = math.hypot(robot_x - f.pos_x, robot_y - f.pos_y)
            if dist < self._r_clear:
                to_remove.append(fid)
                self._cleared_positions.append((f.pos_x, f.pos_y, f.pos_z))

        for fid in to_remove:
            del self._active_frontiers[fid]

    def select_best_frontier(
        self,
        robot_x: float,
        robot_y: float
    ) -> Optional[ManagedFrontier]:
        """Select frontier maximizing global utility u_i = g_i / (1 + distance)."""
        if not self._active_frontiers:
            return None

        best_frontier: Optional[ManagedFrontier] = None
        best_score = -float("inf")

        for f in self._active_frontiers.values():
            dist = math.hypot(robot_x - f.pos_x, robot_y - f.pos_y)
            base_gain = f.exploration_gain * f.semantic_prob
            if f.is_viewpoint_frontier:
                utility = 1000.0 / (1.0 + dist)
            else:
                utility = base_gain / (1.0 + dist)

            f.total_utility = utility
            if utility > best_score:
                best_score = utility
                best_frontier = f

        return best_frontier

    def handle_progress(
        self,
        current_distance: float
    ) -> bool:
        """Check progress against stall threshold; return True if target should be dropped."""
        prog = self._progress
        if prog.d_prev - current_distance >= prog.epsilon_progress_m:
            prog.stall_counter = 0
        else:
            prog.stall_counter += 1

        prog.d_prev = current_distance
        if prog.stall_counter >= prog.stall_cycles_threshold:
            prog.stall_counter = 0
            prog.d_prev = 1e9
            return True
        return False

    def drop_frontier(self, frontier_id: str) -> None:
        """Remove frontier and add its coordinates to cleared set."""
        if frontier_id in self._active_frontiers:
            f = self._active_frontiers.pop(frontier_id)
            self._cleared_positions.append((f.pos_x, f.pos_y, f.pos_z))

    def insert_viewpoint_frontier(
        self,
        target_x: float,
        target_y: float,
        robot_x: float,
        robot_y: float,
        standoff_m: float = 0.8
    ) -> ManagedFrontier:
        """Insert high-priority viewpoint frontier facing the candidate target."""
        dx = target_x - robot_x
        dy = target_y - robot_y
        dist = math.hypot(dx, dy)
        if dist > 1e-3:
            unit_x, unit_y = dx / dist, dy / dist
        else:
            unit_x, unit_y = 1.0, 0.0

        vp_x = target_x - standoff_m * unit_x
        vp_y = target_y - standoff_m * unit_y

        fid = f"VP_{self.generate_next_id()}"
        vp_frontier = ManagedFrontier(
            id=fid,
            pos_x=vp_x,
            pos_y=vp_y,
            pos_z=0.0,
            pixel_u=320,
            pixel_v=240,
            exploration_gain=50.0,
            semantic_prob=1.0,
            total_utility=1000.0,
            is_viewpoint_frontier=True
        )
        self._active_frontiers[fid] = vp_frontier
        return vp_frontier

    def _is_near_cleared(self, x: float, y: float, z: float) -> bool:
        """Return True if point is within r_clear of any cleared position."""
        for cx, cy, cz in self._cleared_positions:
            if math.dist((x, y, z), (cx, cy, cz)) < self._r_clear:
                return True
        return False
