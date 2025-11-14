from heapq import heappop, heappush
from typing import List, Tuple, Optional, Set

from type import OccupancyGrid, Node2D

import math

GridCoord = Tuple[int, int]

class AStar:
  def __init__(self):
    self._map: Optional[OccupancyGrid] = None
    self._path: List[GridCoord] = []
    self._world_path: List[Tuple[float, float, float]] = []
    self._footprint: Optional[Tuple[float, float]] = None
    self._footprint_radius_cells: int = 0
    self._use_footprint_checks: bool = True

  def _get_neighbors(self) -> List[Tuple[int, int, float]]:
    """4-connectivity + diagonals with their respective move costs."""
    return [
      (1, 0, 1.0),
      (0, 1, 1.0),
      (-1, 0, 1.0),
      (0, -1, 1.0),
      (1, 1, math.sqrt(2)),
      (1, -1, math.sqrt(2)),
      (-1, 1, math.sqrt(2)),
      (-1, -1, math.sqrt(2)),
    ]

  def _grid_from_world(self, wx: float, wy: float) -> GridCoord:
    """Convert world coordinates to grid indices."""
    origin = self._map.origin
    res = self._map.resolution
    return (
      math.floor((wx - origin.x) / res),
      math.floor((wy - origin.y) / res),
    )

  def _in_bounds(self, x: int, y: int) -> bool:
    return 0 <= x < self._map.width and 0 <= y < self._map.height

  def set_footprint(self, length: float, width: float) -> None:
    self._footprint = (length, width)
    self._update_footprint_radius()

  def _update_footprint_radius(self) -> None:
    if self._map is None or self._footprint is None:
      self._footprint_radius_cells = 0
      return
    length, width = self._footprint
    radius = 0.5 * math.hypot(length, width)
    self._footprint_radius_cells = max(0, int(math.ceil(radius / self._map.resolution)))

  def _is_cell_free(self, x: int, y: int, threshold: float = 0.15) -> bool:
    if not self._in_bounds(x, y):
      return False
    if not self._use_footprint_checks or self._footprint is None:
      idx = self._map.get_grid_index(x, y)
      value = self._map.data[idx]
      return 0 <= value <= threshold

    radius = self._footprint_radius_cells
    if radius <= 0:
      radius = 0

    for dx in range(-radius, radius + 1):
      for dy in range(-radius, radius + 1):
        gx = x + dx
        gy = y + dy
        if not self._in_bounds(gx, gy):
          return False
        idx = self._map.get_grid_index(gx, gy)
        value = self._map.data[idx]
        if value < 0 or value > threshold:
          return False
    return True

  def plan(self, start: tuple, goal: tuple) -> bool:
    if self._map is None:
      return False

    grid_sx, grid_sy = self.map.map_to_grid(start[0], start[1])
    grid_gx, grid_gy = self.map.map_to_grid(goal[0], goal[1])

    start_node = Node2D(x=grid_sx, y=grid_sy, g=0.0)
    goal_node = Node2D(x=grid_gx, y=grid_gy, g=0.0)

    open_heap: List[Node2D] = []
    heappush(open_heap, start_node)

  def plan(self, start: tuple, goal: tuple, use_footprint: bool = True) -> bool:
    if self._map is None:
      return False

    self._use_footprint_checks = use_footprint

    start_coord = self._grid_from_world(*start)
    goal_coord = self._grid_from_world(*goal)

    if not self._in_bounds(*start_coord) or not self._in_bounds(*goal_coord):
      return False
    if not self._is_cell_free(*start_coord) or not self._is_cell_free(*goal_coord):
      return False

    start_node = Node2D(x=start_coord[0], y=start_coord[1], g=0.0)
    goal_node = Node2D(x=goal_coord[0], y=goal_coord[1], g=0.0)
    start_node.cost(goal_node)

    open_heap: List[Tuple[float, float, int, Node2D]] = []
    counter = 0
    heappush(open_heap, (start_node.f, start_node.h, counter, start_node))

    g_score: dict[GridCoord, float] = {start_coord: 0.0}
    closed: Set[GridCoord] = set()

    while open_heap:
      _, _, _, current = heappop(open_heap)
      current_coord = (current.x, current.y)

      if current_coord in closed:
        continue
      closed.add(current_coord)

      if current == goal_node:
        self._reconstruct_path(current)
        return True

      for dx, dy, move_cost in self._get_neighbors():
        nx = current.x + dx
        ny = current.y + dy
        neighbor_coord = (nx, ny)

        if not self._in_bounds(nx, ny):
          continue
        if not self._is_cell_free(nx, ny):
          continue
        if neighbor_coord in closed:
          continue

        tentative_g = current.g + move_cost
        if tentative_g >= g_score.get(neighbor_coord, math.inf):
          continue

        neighbor = Node2D(x=nx, y=ny, parent=current, g=tentative_g)
        neighbor.cost(goal_node)
        g_score[neighbor_coord] = tentative_g
        counter += 1
        heappush(open_heap, (neighbor.f, neighbor.h, counter, neighbor))

    return False

  def _reconstruct_path(self, node: Node2D) -> None:
    self._path.clear()
    self._world_path.clear()
    current = node
    while current is not None:
      self._path.append((current.x, current.y))
      if self._map is not None:
        wx, wy = self._grid_to_world(current.x, current.y)
        self._world_path.append((wx, wy, 0.0))
      current = current.parent
    self._path.reverse()
    self._world_path.reverse()

  def _grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
    origin = self._map.origin
    res = self._map.resolution
    return origin.x + (gx + 0.5) * res, origin.y + (gy + 0.5) * res

  @property
  def path(self):
    return self._path

  @property
  def world_path(self) -> List[Tuple[float, float, float]]:
    return self._world_path

  @property
  def map(self) -> OccupancyGrid:
    return self._map

  @map.setter
  def map(self, map):
    self._map = map
    self._update_footprint_radius()
