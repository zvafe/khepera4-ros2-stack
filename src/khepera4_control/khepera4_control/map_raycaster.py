# Copyright 2026 Andrei Berteanu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Map-referenced 2D Bresenham occlusion raycaster for occupancy grids."""

import math
from typing import List, Optional, Sequence, Tuple, Union

from nav_msgs.msg import OccupancyGrid


def bresenham_cells_2d(
    start_c: int,
    start_r: int,
    target_c: int,
    target_r: int,
) -> List[Tuple[int, int]]:
    """Compute grid coordinates along a 2D line segment using Bresenham's algorithm.

    :param start_c: Starting column index.
    :param start_r: Starting row index.
    :param target_c: Target column index.
    :param target_r: Target row index.
    :return: List of (col, row) grid index tuples along the line from start to target.
    """
    cells: List[Tuple[int, int]] = []
    dc = abs(target_c - start_c)
    dr = abs(target_r - start_r)
    sc = 1 if start_c < target_c else -1
    sr = 1 if start_r < target_r else -1
    err = dc - dr

    curr_c = start_c
    curr_r = start_r

    while True:
        cells.append((curr_c, curr_r))
        if curr_c == target_c and curr_r == target_r:
            break
        err_doubled = 2 * err
        if err_doubled > -dr:
            err -= dr
            curr_c += sc
        if err_doubled < dc:
            err += dc
            curr_r += sr

    return cells


class MapRaycaster:
    """Raycasts 2D lines across occupancy grids to test line-of-sight occlusion."""

    def __init__(
        self,
        default_occ_threshold: int = 50,
        robot_radius: float = 0.07,
        target_radius: float = 0.06,
        unknown_as_obstacle: bool = False,
    ) -> None:
        """Initialize raycaster with geometric tolerances and occupancy cutoff.

        :param default_occ_threshold: Occupancy cutoff (0..100) above which a cell blocks LOS.
        :param robot_radius: Distance (meters) from origin to ignore (robot chassis margin).
        :param target_radius: Distance (meters) from destination to ignore (target extents).
        :param unknown_as_obstacle: Whether unobserved cells (-1) should be treated as obstacles.
        """
        self.default_occ_threshold = default_occ_threshold
        self.robot_radius = robot_radius
        self.target_radius = target_radius
        self.unknown_as_obstacle = unknown_as_obstacle

    @staticmethod
    def world_to_grid(
        world_x: float,
        world_y: float,
        origin_x: float,
        origin_y: float,
        resolution: float,
    ) -> Tuple[int, int]:
        """Convert continuous 2D world coordinates into integer grid indices (col, row).

        :param world_x: World coordinate along X (meters).
        :param world_y: World coordinate along Y (meters).
        :param origin_x: World coordinate of map origin cell [0, 0] along X (meters).
        :param origin_y: World coordinate of map origin cell [0, 0] along Y (meters).
        :param resolution: Grid cell resolution (meters/cell).
        :return: (col, row) integer grid coordinates.
        """
        col = int(math.floor((world_x - origin_x) / resolution))
        row = int(math.floor((world_y - origin_y) / resolution))
        return col, row

    @staticmethod
    def grid_to_world(
        col: int,
        row: int,
        origin_x: float,
        origin_y: float,
        resolution: float,
    ) -> Tuple[float, float]:
        """Convert grid cell indices to continuous center coordinates in world frame.

        :param col: Grid column index.
        :param row: Grid row index.
        :param origin_x: Map origin X in meters.
        :param origin_y: Map origin Y in meters.
        :param resolution: Grid cell resolution (meters/cell).
        :return: (world_x, world_y) cell center in meters.
        """
        world_x = origin_x + (float(col) + 0.5) * resolution
        world_y = origin_y + (float(row) + 0.5) * resolution
        return world_x, world_y

    def is_line_of_sight_occluded(
        self,
        grid: Union[OccupancyGrid, Sequence[int]],
        start_x: float,
        start_y: float,
        target_x: float,
        target_y: float,
        origin_x: Optional[float] = None,
        origin_y: Optional[float] = None,
        resolution: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        occ_threshold: Optional[int] = None,
        unknown_as_obstacle: Optional[bool] = None,
    ) -> bool:
        """Determine whether direct line of sight between start and target is occluded.

        :param grid: Either an OccupancyGrid message or flat sequence of occupancy integers.
        :param start_x: Robot position X in world frame (meters).
        :param start_y: Robot position Y in world frame (meters).
        :param target_x: Target position X in world frame (meters).
        :param target_y: Target position Y in world frame (meters).
        :param origin_x: Map origin X in world frame. Required if grid is not OccupancyGrid.
        :param origin_y: Map origin Y in world frame. Required if grid is not OccupancyGrid.
        :param resolution: Map resolution. Required if grid is not OccupancyGrid.
        :param width: Map width in cells. Required if grid is not OccupancyGrid.
        :param height: Map height in cells. Required if grid is not OccupancyGrid.
        :param occ_threshold: Occupancy cutoff (default: self.default_occ_threshold).
        :param unknown_as_obstacle: If true, treats -1 as an obstacle.
        :return: True if line of sight is obstructed by an obstacle cell, False otherwise.
        """
        threshold = self.default_occ_threshold if occ_threshold is None else occ_threshold
        treat_unknown_as_obs = (
            self.unknown_as_obstacle
            if unknown_as_obstacle is None
            else unknown_as_obstacle
        )

        if isinstance(grid, OccupancyGrid):
            data = grid.data
            map_res = grid.info.resolution
            map_w = grid.info.width
            map_h = grid.info.height
            map_ox = grid.info.origin.position.x
            map_oy = grid.info.origin.position.y
        else:
            data = grid
            if any(v is None for v in (origin_x, origin_y, resolution, width, height)):
                raise ValueError("Must provide map metadata when grid is not an OccupancyGrid")
            map_res = float(resolution)  # type: ignore
            map_w = int(width)  # type: ignore
            map_h = int(height)  # type: ignore
            map_ox = float(origin_x)  # type: ignore
            map_oy = float(origin_y)  # type: ignore

        if map_res <= 0.0 or map_w <= 0 or map_h <= 0 or len(data) < (map_w * map_h):
            return False

        c0, r0 = self.world_to_grid(start_x, start_y, map_ox, map_oy, map_res)
        c1, r1 = self.world_to_grid(target_x, target_y, map_ox, map_oy, map_res)

        line_cells = bresenham_cells_2d(c0, r0, c1, r1)

        r_robot_sq = self.robot_radius * self.robot_radius
        r_target_sq = self.target_radius * self.target_radius

        for col, row in line_cells:
            # Check if cell lies within map bounds
            if col < 0 or col >= map_w or row < 0 or row >= map_h:
                if treat_unknown_as_obs:
                    return True
                continue

            # Compute world coordinates of cell center
            cx, cy = self.grid_to_world(col, row, map_ox, map_oy, map_res)

            # Skip cells within robot footprint
            dist_robot_sq = (cx - start_x) ** 2 + (cy - start_y) ** 2
            if dist_robot_sq <= r_robot_sq:
                continue

            # Skip cells within target geometry extent
            dist_target_sq = (cx - target_x) ** 2 + (cy - target_y) ** 2
            if dist_target_sq <= r_target_sq:
                continue

            cell_val = data[row * map_w + col]

            if cell_val >= threshold:
                return True
            if treat_unknown_as_obs and cell_val < 0:
                return True

        return False
