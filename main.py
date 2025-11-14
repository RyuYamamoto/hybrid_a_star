import argparse
import math
from typing import Optional

from type import OccupancyGrid, HybridAStarParams, FootPrint

from map_loader import MapLoader
from a_star import AStar
from collision_checker import CollisionChecker
from hybrid_a_star import HybridAStar

from utils import create_footprint_patch

import numpy as np
import matplotlib.pyplot as plt

def visualize(map: OccupancyGrid, start, goal, footprint: FootPrint, path):
  _, ax = plt.subplots(figsize=(8, 6), layout="constrained")

  # plot Occupancy Grid Map
  data = np.asarray(map.data, dtype=np.float32).reshape(map.height, map.width)
  ax.imshow(data, cmap="gray_r", origin="lower")

  # plot start and goal
  sx, sy = map.map_to_grid(start[0], start[1])
  ax.scatter(sx, sy, marker="*", c="r", s=50, label="start")

  for x, y, yaw in path:
    patch = create_footprint_patch((x, y, yaw), footprint, map)
    ax.add_patch(patch)

  ax.grid(True)
  ax.set_title("Hybrid A*")
  plt.legend()
  plt.show()

def main(map_path: str, config: str):
  map_loader = MapLoader(map_path)
  footprint = FootPrint(args.config)
  collision_checker = CollisionChecker()
  collision_checker.map = map_loader.map
  collision_checker.footprint = footprint

  params = HybridAStarParams()
  params.dt = 0.2
  params.v = 0.4
  params.max_steer = math.radians(23.0)
  params.wheel_base = 0.25
  planner = HybridAStar(params, collision_checker)
  planner.map = map_loader.map

  start = (-1.5, -3.0, math.radians(90.0))
  # start = (-3.0, -2.0, math.radians(90.0))
  goal = (2.8, 3.0, math.radians(0.0))
  # goal = (1.0, -7.0, math.radians(-90.0))

  if planner.plan(start, goal) is True:
    visualize(map_loader.map, start, goal, footprint, planner.path)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--map", required=True, help="map path")
  parser.add_argument("--config", required=True)
  args = parser.parse_args()

  main(args.map, args.config)
