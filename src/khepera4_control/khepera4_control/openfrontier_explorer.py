"""
OpenFrontier Map-Based Autonomous Exploration Engine.

Implements visual-language grounded frontier exploration:
- 2D Occupancy Grid Frontier Extraction & Cluster Analysis
- Information Gain Estimation based on Unexplored Volume
- Global Target Management (Merging, Pruning, Selection, Stall Recovery)
- 8-Connected A* Collision-Free Path Planning on Inflated Costmap
- Viewpoint Frontier Prioritization for Target Pursuit Sighting
- Telemetry & Visualization Publishers
"""

from dataclasses import dataclass, field
import heapq
import math
import time
from typing import Dict, List, Optional, Set, Tuple

from geometry_msgs.msg import Point, PointStamped, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
import numpy as np
import scipy.ndimage
from std_msgs.msg import String


@dataclass
class FrontierCandidate:
    """Represents a detected map frontier cluster."""

    id: str
    pos_x: float
    pos_y: float
    cluster_size: int
    exploration_gain: float
    distance: float
    semantic_prob: float = 1.0
    total_utility: float = 0.0
    is_viewpoint_frontier: bool = False
    grid_row: int = 0
    grid_col: int = 0


@dataclass
class StallTracker:
    """Stall detector implementing OpenFrontier Algorithm 3."""

    stall_counter: int = 0
    d_prev: float = 1e9
    stall_cycles_threshold: int = 15
    epsilon_progress_m: float = 0.06


class OpenFrontierExplorer:
    """
    Active map-based frontier exploration engine following the OpenFrontier methodology.

    Extracts frontiers from 2D SLAM occupancy grid, evaluates information gain,
    manages global frontier lifecycle, and plans collision-free paths.
    """

    def __init__(
        self,
        tau_merge_m: float = 0.70,
        r_clear_m: float = 0.35,
        r_near_m: float = 0.30,
        r_goal_m: float = 0.80,
        min_cluster_cells: int = 4,
        robot_safety_radius_m: float = 0.14,
        stall_threshold_cycles: int = 15,
        epsilon_stall_m: float = 0.06,
        arena_extent_m: float = 2.35,
        max_frontier_m: float = 1.55,
    ):
        """Initialize thresholds and tracking structures."""
        self.tau_merge = tau_merge_m
        self.r_clear = r_clear_m
        self.r_near = r_near_m
        self.r_goal = r_goal_m
        self.min_cluster_cells = min_cluster_cells
        self.safety_radius = robot_safety_radius_m
        self.arena_extent = arena_extent_m
        self.max_frontier = max_frontier_m

        self.active_frontiers: Dict[str, FrontierCandidate] = {}
        self.cleared_frontiers: List[Tuple[float, float]] = []
        self.stall_tracker = StallTracker(
            stall_cycles_threshold=stall_threshold_cycles,
            epsilon_progress_m=epsilon_stall_m,
        )
        self._next_id_idx = 0
        self.current_goal: Optional[FrontierCandidate] = None

    def generate_next_id(self) -> str:
        """Generate unique frontier ID label (A, B, ..., Z, F27, ...)."""
        idx = self._next_id_idx
        self._next_id_idx += 1
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if idx < len(chars):
            return chars[idx]
        return f"F{idx}"

    def reset(self) -> None:
        """Clear all active and cleared frontiers."""
        self.active_frontiers.clear()
        self.cleared_frontiers.clear()
        self.stall_tracker = StallTracker()
        self.current_goal = None
        self._next_id_idx = 0

    def detect_frontiers(
        self,
        grid_msg: OccupancyGrid,
        robot_x: float,
        robot_y: float,
        semantic_prior: float = 1.0,
    ) -> List[FrontierCandidate]:
        """
        Extract valid navigable frontier clusters from an OccupancyGrid message.

        A frontier cell is a free cell (0 <= val < 25) that is adjacent (8-connectivity)
        to at least one unknown cell (val == -1), and far enough from obstacles (val >= 50).
        """
        info = grid_msg.info
        w = info.width
        h = info.height
        res = info.resolution
        ox = info.origin.position.x
        oy = info.origin.position.y

        if w < 5 or h < 5 or len(grid_msg.data) != w * h:
            return []

        grid = np.array(grid_msg.data, dtype=np.int8).reshape((h, w))

        # Categorize cells
        free_mask = (grid >= 0) & (grid < 25)
        occ_mask = (grid >= 50)

        # Restrict unknown mask to cells within navigable arena extent
        # to avoid perimeter outside-wall unmappable areas producing phantom frontiers
        col_indices = np.arange(w)
        row_indices = np.arange(h)
        col_x = ox + (col_indices + 0.5) * res
        row_y = oy + (row_indices + 0.5) * res
        valid_x = np.abs(col_x) <= self.arena_extent
        valid_y = np.abs(row_y) <= self.arena_extent
        in_arena_mask = np.outer(valid_y, valid_x)
        unknown_mask = (grid == -1) & in_arena_mask

        if not np.any(free_mask) or not np.any(unknown_mask):
            return []

        # Inflate obstacles with robot safety radius
        inflate_cells = max(2, int(math.ceil(self.safety_radius / res)))
        kernel_size = 2 * inflate_cells + 1
        y, x = np.ogrid[-inflate_cells:inflate_cells + 1, -inflate_cells:inflate_cells + 1]
        disk_kernel = (x * x + y * y <= inflate_cells * inflate_cells)

        inflated_occ = scipy.ndimage.binary_dilation(occ_mask, structure=disk_kernel)
        safe_free = free_mask & (~inflated_occ)

        if not np.any(safe_free):
            return []

        # Find free cells adjacent to unknown space
        unknown_border = scipy.ndimage.binary_dilation(unknown_mask, structure=np.ones((3, 3)))
        frontier_mask = safe_free & unknown_border

        if not np.any(frontier_mask):
            return []

        # Connected component clustering with 8-connectivity
        labeled_mask, num_clusters = scipy.ndimage.label(
            frontier_mask, structure=np.ones((3, 3))
        )
        if num_clusters == 0:
            return []

        proposals: List[FrontierCandidate] = []
        neighborhood_radius = max(5, int(1.0 / res))

        for c_id in range(1, num_clusters + 1):
            cluster_cells = np.argwhere(labeled_mask == c_id)
            c_size = len(cluster_cells)
            if c_size < self.min_cluster_cells:
                continue

            # Centroid in grid coordinates
            mean_row = float(np.mean(cluster_cells[:, 0]))
            mean_col = float(np.mean(cluster_cells[:, 1]))

            # Find cell in cluster closest to mean that is in safe_free
            dists = np.sum((cluster_cells - np.array([mean_row, mean_col])) ** 2, axis=1)
            best_cell_idx = int(np.argmin(dists))
            best_row, best_col = cluster_cells[best_cell_idx]

            # Convert to world meters
            wx = float(ox + (best_col + 0.5) * res)
            wy = float(oy + (best_row + 0.5) * res)

            # Skip if on or beyond outer perimeter walls
            if abs(wx) > self.arena_extent or abs(wy) > self.arena_extent:
                continue

            # Bound frontier targets to safe inner perimeter so the robot maintains
            # full speed with clear sensor vantage without grazing perimeter walls
            wx = float(np.clip(wx, -self.max_frontier, self.max_frontier))
            wy = float(np.clip(wy, -self.max_frontier, self.max_frontier))
            best_col = min(w - 1, max(0, int(math.floor((wx - ox) / res))))
            best_row = min(h - 1, max(0, int(math.floor((wy - oy) / res))))

            # Skip if near cleared/visited frontier
            if self._is_near_cleared(wx, wy):
                continue


            # Information Gain: cluster length + count of unknown cells in local neighborhood
            r0 = max(0, best_row - neighborhood_radius)
            r1 = min(h, best_row + neighborhood_radius + 1)
            c0 = max(0, best_col - neighborhood_radius)
            c1 = min(w, best_col + neighborhood_radius + 1)
            unknown_neighborhood = int(np.count_nonzero(unknown_mask[r0:r1, c0:c1]))

            gain = float(c_size * res * 20.0 + unknown_neighborhood * 0.15)
            dist_to_robot = float(math.hypot(wx - robot_x, wy - robot_y))

            # OpenFrontier utility formula: u_i = (p_i * g_i) / max(d_i, 0.25)
            utility = float((semantic_prior * gain) / max(dist_to_robot, 0.25))

            fid = self.generate_next_id()
            proposals.append(
                FrontierCandidate(
                    id=fid,
                    pos_x=wx,
                    pos_y=wy,
                    cluster_size=c_size,
                    exploration_gain=gain,
                    distance=dist_to_robot,
                    semantic_prob=semantic_prior,
                    total_utility=utility,
                    is_viewpoint_frontier=False,
                    grid_row=int(best_row),
                    grid_col=int(best_col),
                )
            )

        return proposals

    def merge_proposals(self, proposals: List[FrontierCandidate]) -> None:
        """Merge newly extracted frontiers into active set within spatial tolerance tau_merge."""
        for prop in proposals:
            if self._is_near_cleared(prop.pos_x, prop.pos_y):
                continue

            matched_id: Optional[str] = None
            for fid, existing in self.active_frontiers.items():
                d = math.hypot(prop.pos_x - existing.pos_x, prop.pos_y - existing.pos_y)
                if d < self.tau_merge:
                    matched_id = fid
                    break

            if matched_id is not None:
                existing = self.active_frontiers[matched_id]
                existing.pos_x = 0.7 * existing.pos_x + 0.3 * prop.pos_x
                existing.pos_y = 0.7 * existing.pos_y + 0.3 * prop.pos_y
                existing.exploration_gain = max(existing.exploration_gain, prop.exploration_gain)
                existing.cluster_size = max(existing.cluster_size, prop.cluster_size)
                existing.grid_row = prop.grid_row
                existing.grid_col = prop.grid_col
            else:
                self.active_frontiers[prop.id] = prop

    def prune_cleared_frontiers(self, robot_x: float, robot_y: float) -> None:
        """Remove active frontiers that the robot has approached within r_clear."""
        to_remove: List[str] = []
        for fid, f in self.active_frontiers.items():
            dist = math.hypot(robot_x - f.pos_x, robot_y - f.pos_y)
            if dist < self.r_clear:
                to_remove.append(fid)
                self.cleared_frontiers.append((f.pos_x, f.pos_y))

        for fid in to_remove:
            del self.active_frontiers[fid]

    def select_best_frontier(
        self,
        robot_x: float,
        robot_y: float,
        visited_history: Optional[List[Tuple[float, float]]] = None,
    ) -> Optional[FrontierCandidate]:
        """Select frontier maximizing global utility u_i = g_i / (1.0 + 0.35 * d_i) - penalty."""
        if not self.active_frontiers:
            return None

        best_frontier: Optional[FrontierCandidate] = None
        best_score = -float("inf")

        for f in self.active_frontiers.values():
            dist = float(math.hypot(robot_x - f.pos_x, robot_y - f.pos_y))
            f.distance = dist

            if f.is_viewpoint_frontier:
                utility = float(1000.0 / max(dist, 0.20))
            else:
                gain = f.exploration_gain * f.semantic_prob
                visited_penalty = 0.0
                if visited_history:
                    for vx, vy in visited_history:
                        d_vis = math.hypot(f.pos_x - vx, f.pos_y - vy)
                        if d_vis < 0.90:
                            visited_penalty += 80.0 * (1.0 - d_vis / 0.90)
                utility = float(gain / (1.0 + 0.35 * dist)) - visited_penalty


            f.total_utility = utility
            if utility > best_score:
                best_score = utility
                best_frontier = f

        self.current_goal = best_frontier
        return best_frontier


    def handle_progress(self, current_distance: float) -> bool:
        """Check progress against stall threshold (Algorithm 3). Returns True if target should be dropped."""
        prog = self.stall_tracker
        if prog.d_prev - current_distance >= prog.epsilon_progress_m:
            prog.stall_counter = 0
            prog.d_prev = current_distance
        else:
            prog.stall_counter += 1

        if prog.stall_counter >= prog.stall_cycles_threshold:
            prog.stall_counter = 0
            prog.d_prev = 1e9
            return True
        return False

    def drop_frontier(self, frontier_id: str) -> None:
        """Remove frontier and add its coordinates to cleared set."""
        if frontier_id in self.active_frontiers:
            f = self.active_frontiers.pop(frontier_id)
            self.cleared_frontiers.append((f.pos_x, f.pos_y))
        if self.current_goal and self.current_goal.id == frontier_id:
            self.current_goal = None

    def insert_viewpoint_frontier(
        self,
        target_x: float,
        target_y: float,
        robot_x: float,
        robot_y: float,
        standoff_m: float = 0.70,
    ) -> FrontierCandidate:
        """Instantiate high-utility viewpoint frontier facing target (Algorithm 2)."""
        dx = target_x - robot_x
        dy = target_y - robot_y
        dist = math.hypot(dx, dy)
        if dist > 1e-3:
            unit_x, unit_y = dx / dist, dy / dist
        else:
            unit_x, unit_y = 1.0, 0.0

        vp_x = float(target_x - standoff_m * unit_x)
        vp_y = float(target_y - standoff_m * unit_y)
        fid = f"VP_{self.generate_next_id()}"

        vp_frontier = FrontierCandidate(
            id=fid,
            pos_x=vp_x,
            pos_y=vp_y,
            cluster_size=50,
            exploration_gain=100.0,
            distance=float(math.hypot(vp_x - robot_x, vp_y - robot_y)),
            semantic_prob=1.0,
            total_utility=1000.0,
            is_viewpoint_frontier=True,
        )
        self.active_frontiers[fid] = vp_frontier
        self.current_goal = vp_frontier
        return vp_frontier

    def plan_path(
        self,
        grid_msg: OccupancyGrid,
        start_x: float,
        start_y: float,
        goal_x: float,
        goal_y: float,
    ) -> List[Tuple[float, float]]:
        """
        Compute an 8-connected A* collision-free path from (start_x, start_y) to (goal_x, goal_y).

        Returns smoothed list of waypoint tuples [(x0, y0), (x1, y1), ...].
        """
        info = grid_msg.info
        w = info.width
        h = info.height
        res = info.resolution
        ox = info.origin.position.x
        oy = info.origin.position.y

        grid = np.array(grid_msg.data, dtype=np.int8).reshape((h, w))
        # For exploration planning: navigable space includes both known free cells (0..24)
        # and unexplored cells (-1) assumed navigable, strictly excluding obstacles (>= 50)
        navigable_mask = (grid < 50)
        occ_mask = (grid >= 50)

        # Inflate obstacles with safety margin
        inflate_cells = max(1, int(math.ceil(self.safety_radius / res)))
        kernel_size = 2 * inflate_cells + 1
        y, x = np.ogrid[-inflate_cells:inflate_cells + 1, -inflate_cells:inflate_cells + 1]
        disk_kernel = (x * x + y * y <= inflate_cells * inflate_cells)
        inflated_occ = scipy.ndimage.binary_dilation(occ_mask, structure=disk_kernel)
        safe_grid = navigable_mask & (~inflated_occ)

        # Start and Goal grid coordinates
        s_col = int(math.floor((start_x - ox) / res))
        s_row = int(math.floor((start_y - oy) / res))
        g_col = int(math.floor((goal_x - ox) / res))
        g_row = int(math.floor((goal_y - oy) / res))

        # Clamp to bounds
        s_col = min(w - 1, max(0, s_col))
        s_row = min(h - 1, max(0, s_row))
        g_col = min(w - 1, max(0, g_col))
        g_row = min(h - 1, max(0, g_row))

        # If start is in obstacle, find nearest safe free cell
        if not safe_grid[s_row, s_col]:
            s_row, s_col = self._find_nearest_walkable(safe_grid, s_row, s_col)

        # If goal is in obstacle or border, find nearest safe free cell
        if not safe_grid[g_row, g_col]:
            g_row, g_col = self._find_nearest_walkable(safe_grid, g_row, g_col)

        if s_row == g_row and s_col == g_col:
            return [(goal_x, goal_y)]

        # 8-connected A* search
        start = (s_row, s_col)
        goal = (g_row, g_col)

        motions = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414),
        ]

        def heuristic(r1: int, c1: int, r2: int, c2: int) -> float:
            dr = abs(r1 - r2)
            dc = abs(c1 - c2)
            return float((dr + dc) + (1.414 - 2.0) * min(dr, dc))

        open_set: List[Tuple[float, float, int, int]] = []
        heapq.heappush(open_set, (heuristic(s_row, s_col, g_row, g_col), 0.0, s_row, s_col))
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_cost: Dict[Tuple[int, int], float] = {start: 0.0}

        max_iterations = 4000
        iterations = 0
        found = False

        while open_set and iterations < max_iterations:
            iterations += 1
            _, current_g, r, c = heapq.heappop(open_set)

            if (r, c) == goal:
                found = True
                break

            if current_g > g_cost.get((r, c), float("inf")):
                continue

            for dr, dc, cost in motions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and safe_grid[nr, nc]:
                    tentative_g = current_g + cost
                    if tentative_g < g_cost.get((nr, nc), float("inf")):
                        g_cost[(nr, nc)] = tentative_g
                        priority = tentative_g + heuristic(nr, nc, g_row, g_col)
                        came_from[(nr, nc)] = (r, c)
                        heapq.heappush(open_set, (priority, tentative_g, nr, nc))

        if not found:
            # Fallback to direct goal point if A* reached limits
            return [(goal_x, goal_y)]

        # Reconstruct path
        path_cells: List[Tuple[int, int]] = []
        curr = goal
        while curr in came_from:
            path_cells.append(curr)
            curr = came_from[curr]
        path_cells.append(start)
        path_cells.reverse()

        # Convert to world waypoints and downsample (e.g., every 0.15m ~ 3 cells)
        step = max(1, int(0.15 / res))
        waypoints: List[Tuple[float, float]] = []
        for i in range(0, len(path_cells), step):
            pr, pc = path_cells[i]
            wx = float(ox + (pc + 0.5) * res)
            wy = float(oy + (pr + 0.5) * res)
            waypoints.append((wx, wy))

        # Always include exact goal as final waypoint
        if waypoints and (abs(waypoints[-1][0] - goal_x) > 0.05 or abs(waypoints[-1][1] - goal_y) > 0.05):
            waypoints.append((goal_x, goal_y))

        return waypoints

    def _find_nearest_walkable(
        self,
        safe_grid: np.ndarray,
        r: int,
        c: int,
        search_radius: int = 15,
    ) -> Tuple[int, int]:
        """Find the nearest safe free grid cell within search_radius."""
        h, w = safe_grid.shape
        best_dist = float("inf")
        best_coord = (r, c)

        for dr in range(-search_radius, search_radius + 1):
            for dc in range(-search_radius, search_radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and safe_grid[nr, nc]:
                    d = dr * dr + dc * dc
                    if d < best_dist:
                        best_dist = d
                        best_coord = (nr, nc)

        return best_coord

    def _is_near_cleared(self, x: float, y: float) -> bool:
        """Return True if (x, y) is within r_clear of any cleared position."""
        for cx, cy in self.cleared_frontiers:
            if math.hypot(x - cx, y - cy) < self.r_clear:
                return True
        return False
