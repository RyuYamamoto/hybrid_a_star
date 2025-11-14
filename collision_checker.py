import math
import argparse
from typing import List, Tuple

from type import OccupancyGrid, FootPrint
from utils import create_footprint_patch, transform_footprint
from map_loader import MapLoader

import matplotlib.pyplot as plt

import numpy as np

class CollisionChecker:
  def __init__(self):
    self._footprint: FootPrint = None
    self._map: OccupancyGrid = None

  def _in_polygon(self, x, y, polygon: List[Tuple[float, float]]) -> bool:
    inside = False
    n = len(polygon)

    for i in range(n):
      x0, y0 = polygon[i]
      x1, y1 = polygon[(i + 1) % n]

      if ((y0 > y) != (y1 > y)):
        x_intersect = x0 + (x1 - x0) * (y - y0) / (y1 - y0)
        if x_intersect >= x:
          inside = not inside

    return inside

  def is_collision(self, pose: Tuple[float, ...], cost_threshold: float = 0.15) -> bool:
    if self._map is None or self._footprint is None:
        return False

    fp = self._footprint
    half_len = fp.length / 2.0
    half_wid = fp.width / 2.0

    transformed_world = transform_footprint(footprint=fp, pose=pose)

    xs = [p[0] for p in transformed_world]
    ys = [p[1] for p in transformed_world]
    min_wx, max_wx = min(xs), max(xs)
    min_wy, max_wy = min(ys), max(ys)

    res = self._map.resolution
    ox  = self._map.origin.x
    oy  = self._map.origin.y

    min_gx = int((min_wx - ox) / res) - 1
    max_gx = int((max_wx - ox) / res) + 1
    min_gy = int((min_wy - oy) / res) - 1
    max_gy = int((max_wy - oy) / res) + 1

    min_gx = max(min_gx, 0)
    min_gy = max(min_gy, 0)
    max_gx = min(max_gx, self._map.width  - 1)
    max_gy = min(max_gy, self._map.height - 1)

    get_cost = self._map.get_cost
    px, py, yaw = pose
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    for gy in range(min_gy, max_gy + 1):
      wy = oy + (gy + 0.5) * res
      dy = wy - py

      for gx in range(min_gx, max_gx + 1):
        wx = ox + (gx + 0.5) * res
        dx = wx - px


        x_r =  cos_yaw * dx + sin_yaw * dy
        y_r = -sin_yaw * dx + cos_yaw * dy

        if (abs(x_r) > half_len) or (abs(y_r) > half_wid):
          continue
        cost = get_cost(gx, gy)
        if cost is not None and cost_threshold <= cost:
          return True

    return False

  @property
  def map(self) -> OccupancyGrid:
    return self._map

  @map.setter
  def map(self, grid_map: OccupancyGrid):
    self._map = grid_map

  @property
  def footprint(self) -> FootPrint:
    return self._footprint

  @footprint.setter
  def footprint(self, footprint: FootPrint):
    self._footprint = footprint

def main(
    pose:Tuple[float, ...],
    footprint: FootPrint, map: OccupancyGrid,
    collision_checker: CollisionChecker
  ):
  _, ax = plt.subplots(figsize=(10, 10))

  data = np.asarray(map.data, dtype=np.float32).reshape(map.height, map.width)
  ax.imshow(data, cmap="gray_r", origin="lower")

  s_dx = 5.0 * math.cos(pose[2])
  s_dy = 5.0 * math.sin(pose[2])
  sx, sy = map.map_to_grid(pose[0], pose[1])
  ax.arrow(sx, sy, s_dx, s_dy, head_width=5.0 * 0.5, head_length=5.0 * 0.8, fc="r", ec="r", length_includes_head=True)

  patch = create_footprint_patch(
    pose=pose, footprint=footprint, map=map
  )
  ax.add_patch(patch)

  gx, gy = map.map_to_grid(pose[0], pose[1])

  is_collision = collision_checker.is_collision(pose)
  text = "Collision" if is_collision else "Free"
  color = "red" if is_collision else "blue"

  ax.text(
    gx-40, gy+20, text,
    fontsize=8,
    color=color,
    fontweight="bold",
    va="center",
    ha="left"
  )

  ax.grid(True)
  ax.set_title("Collision Check Test")
  plt.show()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--map", required=True)
  parser.add_argument("--config", required=True)
  parser.add_argument("--x", type=float, required=True)
  parser.add_argument("--y", type=float, required=True)
  parser.add_argument("--yaw", type=float, required=True)
  args = parser.parse_args()

  map_loader = MapLoader(args.map)
  footprint = FootPrint(args.config)

  collision_checker = CollisionChecker()
  collision_checker.map = map_loader.map
  collision_checker.footprint = footprint

  main(
    pose=[args.x, args.y, math.radians(args.yaw)],
    footprint=footprint,
    map=map_loader.map,
    collision_checker=collision_checker
  )
