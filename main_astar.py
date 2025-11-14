import argparse
from typing import Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from map_loader import MapLoader
from a_star import AStar
from type import OccupancyGrid, FootPrint
from utils import create_footprint_patch


def visualize(
  grid_map: OccupancyGrid,
  start: Tuple[float, float],
  goal: Tuple[float, float],
  footprint: Optional[FootPrint],
  path: Optional[List[Tuple[int, int]]] = None,
  world_path: Optional[List[Tuple[float, float, float]]] = None,
) -> None:
  """Simple visualization of the A* solution on the occupancy grid."""
  fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")

  data = np.asarray(grid_map.data, dtype=np.float32).reshape(grid_map.height, grid_map.width)
  ax.imshow(data, cmap="gray_r", origin="lower")

  sx, sy = grid_map.map_to_grid(start[0], start[1])
  gx, gy = grid_map.map_to_grid(goal[0], goal[1])
  ax.scatter(sx, sy, marker="*", c="r", s=50, label="start")
  # ax.scatter(gx, gy, marker="*", c="b", s=50, label="goal")

  if path:
    px, py = zip(*path)
    ax.scatter(px, py, c="lime", s=10, label="path", alpha=0.8)

  if footprint and world_path:
    fp_tuple = (footprint.length, footprint.width)
    for pose in world_path:
      patch: Optional[Polygon] = create_footprint_patch(pose=pose, footprint=footprint, map=grid_map)
      if patch:
        patch.set_edgecolor("black")
        patch.set_facecolor("none")
        # ax.add_patch(patch)

  ax.set_title("A* Path")
  ax.set_xlabel("Grid X")
  ax.set_ylabel("Grid Y")
  ax.set_aspect("equal")
  ax.grid(True)
  ax.legend()
  plt.show()


def main(map_path: str, config: str) -> None:
  map_loader = MapLoader(map_path)
  footprint = FootPrint(config)

  planner = AStar()
  planner.map = map_loader.map
  planner.set_footprint(footprint.length, footprint.width)

  # start = (-2.5, 9.5)
  # goal = (2.5, -9.9)
  start = (-1.5, -3.0)
  goal = (2.8, 3.0)

  if planner.plan(start, goal, use_footprint=False):
    print("A* planning succeeded.")
    visualize(
      grid_map=map_loader.map,
      start=start,
      goal=goal,
      footprint=footprint,
      path=planner.path,
      world_path=planner.world_path,
    )
  else:
    print("A* planning failed.")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--map", required=True, help="Path to the map directory containing map.yaml.")
  parser.add_argument("--config", required=True, help="Vehicle footprint config (JSON).")
  args = parser.parse_args()
  main(args.map, args.config)
