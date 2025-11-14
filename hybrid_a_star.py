import math
import numpy as np
from heapq import heappop, heappush

from typing import Dict, List, Set, Tuple, Optional

from type import OccupancyGrid, HybridNode, HybridAStarParams

from collision_checker import CollisionChecker

class HybridAStar:
  def __init__(self, planner_params: HybridAStarParams, collision_checker: CollisionChecker):
    self._path = []
    self.planner_params = planner_params
    self.collision_checker = collision_checker

  def _prepare_pose(self, pose: Tuple[float, ...]) -> List[float]:
    if len(pose) == 2:
      px, py = pose
      yaw = 0.0
    elif len(pose) == 3:
      px, py, yaw = pose
    else:
      raise ValueError("Pose must be (x, y) or (x, y, yaw)")
    return [px, py, self._normalized(yaw)]

  def plan(self, start: Tuple[float, ...], goal: Tuple[float, ...]) -> bool:
    if self._map is None:
      return False

    start_pose = self._prepare_pose(start)
    goal_pose = self._prepare_pose(goal)

    if self.collision_checker.is_collision(start_pose) or self.collision_checker.is_collision(goal_pose):
      print("Position is in Obstacle.")
      return False

    start_node = HybridNode(start_pose[0], start_pose[1], start_pose[2], g=0.0)
    start_node.cost([goal_pose[0], goal_pose[1]])

    count = 0
    open_list: List[Tuple[float, float, int, HybridNode]] = []
    heappush(open_list, (start_node.f, start_node.h, count, start_node))

    closed: Set[Tuple[int, int, int]] = set()
    g_score: Dict[Tuple[int, int, int], float] = {
      self._discretize(start_pose[0], start_pose[1], start_pose[2]): 0.0
    }

    while open_list:
      _, _, _, current = heappop(open_list)
      key = self._discretize(current.x, current.y, current.yaw)

      if g_score.get(key, math.inf) < current.g:
        continue

      if key in closed:
        continue
      closed.add(key)

      if self._is_goal(current, goal_pose):
        self._reconstruct_path(current)
        return True

      for neighbor in self._node_expand(current):
        neighbor_key = self._discretize(neighbor.x, neighbor.y, neighbor.yaw)
        if neighbor_key in closed:
          continue

        if g_score.get(neighbor_key, math.inf) <= neighbor.g:
          continue

        neighbor.cost((goal_pose[0], goal_pose[1]))
        g_score[neighbor_key] = neighbor.g
        count+=1
        heappush(open_list, (neighbor.f, neighbor.h, count, neighbor))

    return False

  def _discretize(self, x: float, y: float, yaw: float) -> Tuple[int, int, int]:
    gx, gy = self._map.map_to_grid(x, y)
    yaw_idx = int((
     self._normalized(yaw) + math.pi) / (2 * math.pi) * self.planner_params.yaw_discretization
    )
    yaw_idx = max(0, min(self.planner_params.yaw_discretization - 1, yaw_idx))
    return gx, gy, yaw_idx

  def _node_expand(self, node: HybridNode) -> List[HybridNode]:
    neighbors: List[HybridNode] = []

    max_steer = self.planner_params.max_steer
    v = self.planner_params.v
    dt = self.planner_params.dt
    wheel_base = self.planner_params.wheel_base
    for steer in np.linspace(-max_steer, max_steer, 7):
      new_yaw = self._normalized(node.yaw + (v *dt * math.tan(steer)) / wheel_base)
      new_x = node.x + v * math.cos(new_yaw) * dt
      new_y = node.y + v * math.sin(new_yaw) * dt

      nx, ny = self._map.map_to_grid(new_x, new_y)
      is_bounds = (0 <= nx < self._map.width and 0 <= ny < self._map.height)
      if is_bounds is False:
        continue

      if self.collision_checker.is_collision([new_x, new_y, new_yaw]):
        continue
      # if not self._is_cell_free(new_x, new_y):
        # continue

      obstacle_cost = self._map.get_cost(nx, ny)

      cost = node.g + abs(v) * dt + 1.0 * abs(steer) + 1.0 * obstacle_cost
      neighbors.append(
        HybridNode(new_x, new_y, new_yaw, g=cost, parent=node)
      )

    return neighbors

  def _reconstruct_path(self, node: HybridNode) -> None:
    self._path.clear()
    current = node
    while current:
      self._path.append((current.x, current.y, current.yaw))
      current = current.parent
    self._path.reverse()

  def _is_goal(self, node: HybridNode, goal: Tuple[float, float, float]):
    dist = math.hypot(node.x - goal[0], node.y - goal[1])
    if self.planner_params.goal_xy_tolerance < dist:
      return False
    return abs(self._normalized(node.yaw - goal[2])) <= self.planner_params.goal_yaw_tolerance

  def _normalized(self, angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

  @property
  def map(self) -> Optional[OccupancyGrid]:
    return self._map

  @map.setter
  def map(self, map: OccupancyGrid):
    self._map = map

  @property
  def path(self) -> List[Tuple[float, ...]]:
    return self._path

  @property
  def world_path(self) -> List[Tuple[float, ...]]:
    return self._world_path
