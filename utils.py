from typing import List, Tuple
import numpy as np

import math
from matplotlib.patches import Polygon

from type import FootPrint, OccupancyGrid

def create_footprint_patch(pose: Tuple[float, ...], footprint: FootPrint, map: OccupancyGrid):
  c, s = np.cos(pose[2]), np.sin(pose[2])
  rot = np.array([[c, -s], [s, c]], dtype=np.float32)
  rotated = np.array(footprint.footprint) @ rot.T + [pose[0], pose[1]]

  res = map.resolution
  origin = map.origin
  grid_pos = [
    ((x - origin.x) / res, (y - origin.y) / res)
    for x, y in rotated
  ]

  return Polygon(
    grid_pos,
    closed=True,
    facecolor="y",
    edgecolor="k",
    linewidth=1.0,
    alpha=0.5
  )

def transform_footprint(
    footprint: FootPrint, pose: Tuple[float, ...]
  ) -> List[Tuple[float, float]]:
  transformed = []
  for px, py in footprint.footprint:
    tx = pose[0] + math.cos(pose[2]) * px - math.sin(pose[2]) * py
    ty = pose[1] + math.sin(pose[2]) * px + math.cos(pose[2]) * py
    transformed.append((tx, ty))

  return transformed
