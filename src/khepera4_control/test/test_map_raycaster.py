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

"""Unit tests for MapRaycaster 2D Bresenham occlusion detection."""

from geometry_msgs.msg import Point, Pose
from nav_msgs.msg import MapMetaData, OccupancyGrid
import pytest

from khepera4_control.map_raycaster import MapRaycaster, bresenham_cells_2d


def test_bresenham_cells_horizontal_vertical():
    """Verify Bresenham algorithm on exact horizontal and vertical lines."""
    h_cells = bresenham_cells_2d(0, 0, 3, 0)
    assert h_cells == [(0, 0), (1, 0), (2, 0), (3, 0)]

    v_cells = bresenham_cells_2d(2, 1, 2, 4)
    assert v_cells == [(2, 1), (2, 2), (2, 3), (2, 4)]

    rev_cells = bresenham_cells_2d(3, 0, 0, 0)
    assert rev_cells == [(3, 0), (2, 0), (1, 0), (0, 0)]


def test_bresenham_cells_diagonal_and_single_point():
    """Verify Bresenham algorithm on diagonal line and identity point."""
    diag_cells = bresenham_cells_2d(0, 0, 2, 2)
    assert diag_cells == [(0, 0), (1, 1), (2, 2)]

    single_cell = bresenham_cells_2d(5, 5, 5, 5)
    assert single_cell == [(5, 5)]


def _create_mock_occupancy_grid(
    width: int,
    height: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    default_val: int = 0,
) -> OccupancyGrid:
    """Create test OccupancyGrid message with uniform default value."""
    grid = OccupancyGrid()
    grid.info = MapMetaData()
    grid.info.width = width
    grid.info.height = height
    grid.info.resolution = resolution
    grid.info.origin = Pose()
    grid.info.origin.position = Point(x=origin_x, y=origin_y, z=0.0)
    grid.data = [default_val] * (width * height)
    return grid


def test_clear_line_of_sight():
    """Verify clear line of sight returns False on open floor."""
    grid = _create_mock_occupancy_grid(
        width=50,
        height=50,
        resolution=0.05,
        origin_x=-1.25,
        origin_y=-1.25,
        default_val=0,
    )
    raycaster = MapRaycaster(default_occ_threshold=50)

    # Line of sight from (0.0, 0.0) to (1.0, 0.0)
    is_blocked = raycaster.is_line_of_sight_occluded(
        grid=grid,
        start_x=0.0,
        start_y=0.0,
        target_x=1.0,
        target_y=0.0,
    )
    assert not is_blocked


def test_occluded_by_wall():
    """Verify wall placed in between start and target blocks line of sight."""
    width, height = 50, 50
    res = 0.05
    ox, oy = -1.25, -1.25
    grid = _create_mock_occupancy_grid(
        width=width,
        height=height,
        resolution=res,
        origin_x=ox,
        origin_y=oy,
        default_val=0,
    )

    # Place wall cell at world (0.50, 0.0)
    c_wall, r_wall = MapRaycaster.world_to_grid(0.50, 0.0, ox, oy, res)
    grid.data[r_wall * width + c_wall] = 100

    raycaster = MapRaycaster(default_occ_threshold=50)
    is_blocked = raycaster.is_line_of_sight_occluded(
        grid=grid,
        start_x=0.0,
        start_y=0.0,
        target_x=1.0,
        target_y=0.0,
    )
    assert is_blocked


def test_obstacle_within_robot_chassis_ignored():
    """Verify obstacle inside robot radius margin does not falsely occlude."""
    width, height = 50, 50
    res = 0.05
    ox, oy = -1.25, -1.25
    grid = _create_mock_occupancy_grid(
        width=width,
        height=height,
        resolution=res,
        origin_x=ox,
        origin_y=oy,
        default_val=0,
    )

    # Place obstacle cell at world (0.03, 0.0) - well within robot_radius=0.07m
    c_close, r_close = MapRaycaster.world_to_grid(0.03, 0.0, ox, oy, res)
    grid.data[r_close * width + c_close] = 100

    raycaster = MapRaycaster(default_occ_threshold=50, robot_radius=0.07)
    is_blocked = raycaster.is_line_of_sight_occluded(
        grid=grid,
        start_x=0.0,
        start_y=0.0,
        target_x=1.0,
        target_y=0.0,
    )
    assert not is_blocked


def test_obstacle_within_target_extent_ignored():
    """Verify obstacle inside target bounding margin does not falsely occlude."""
    width, height = 50, 50
    res = 0.05
    ox, oy = -1.25, -1.25
    grid = _create_mock_occupancy_grid(
        width=width,
        height=height,
        resolution=res,
        origin_x=ox,
        origin_y=oy,
        default_val=0,
    )

    # Target is at (1.0, 0.0). Place obstacle at (0.97, 0.0) - within target_radius=0.06m
    c_target_box, r_target_box = MapRaycaster.world_to_grid(0.97, 0.0, ox, oy, res)
    grid.data[r_target_box * width + c_target_box] = 100

    raycaster = MapRaycaster(default_occ_threshold=50, target_radius=0.06)
    is_blocked = raycaster.is_line_of_sight_occluded(
        grid=grid,
        start_x=0.0,
        start_y=0.0,
        target_x=1.0,
        target_y=0.0,
    )
    assert not is_blocked


def test_obstacle_behind_target_ignored():
    """Verify obstacle positioned past the target does not affect LOS."""
    width, height = 50, 50
    res = 0.05
    ox, oy = -1.25, -1.25
    grid = _create_mock_occupancy_grid(
        width=width,
        height=height,
        resolution=res,
        origin_x=ox,
        origin_y=oy,
        default_val=0,
    )

    # Place wall behind target at (1.20, 0.0)
    c_behind, r_behind = MapRaycaster.world_to_grid(1.20, 0.0, ox, oy, res)
    grid.data[r_behind * width + c_behind] = 100

    raycaster = MapRaycaster(default_occ_threshold=50)
    is_blocked = raycaster.is_line_of_sight_occluded(
        grid=grid,
        start_x=0.0,
        start_y=0.0,
        target_x=0.80,
        target_y=0.0,
    )
    assert not is_blocked


def test_raw_list_input_and_invalid_metadata():
    """Verify MapRaycaster supports raw sequence inputs and validates metadata."""
    raycaster = MapRaycaster()
    raw_data = [0] * 100
    is_blocked = raycaster.is_line_of_sight_occluded(
        grid=raw_data,
        start_x=0.0,
        start_y=0.0,
        target_x=0.5,
        target_y=0.0,
        origin_x=-0.25,
        origin_y=-0.25,
        resolution=0.05,
        width=10,
        height=10,
    )
    assert not is_blocked

    with pytest.raises(ValueError):
        raycaster.is_line_of_sight_occluded(
            grid=raw_data,
            start_x=0.0,
            start_y=0.0,
            target_x=0.5,
            target_y=0.0,
        )
