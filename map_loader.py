import yaml
import os
import numpy as np
from PIL import Image
from typing import Optional

from type import OccupancyGrid, Pose2D

class MapLoader:
  def __init__(self, map_path: str):
    self.map_path = map_path
    self._map = self.load_map(self.map_path)

  def load_map(self, map_path: str) -> OccupancyGrid:
    map_yaml_path = os.path.join(map_path, "map.yaml")
    with open(map_yaml_path, "r") as f:
      map_config = yaml.safe_load(f)

    # Initialized Occupancy Grid
    grid_map = OccupancyGrid()
    grid_map.resolution = map_config["resolution"]
    grid_map.origin = Pose2D(map_config["origin"][0],map_config["origin"][1], 0.0)

    # Load map pgm
    map_pgm_path = os.path.join(map_path, map_config["image"])
    map_pgm = Image.open(map_pgm_path)
    map_pgm_data = np.array(map_pgm)

    grid_map.height, grid_map.width = map_pgm_data.shape

    probability = np.zeros_like(map_pgm_data, dtype=np.float32)
    probability = (255 - map_pgm_data.astype(np.float32)) / 255.0
    grid_map.data = probability.flatten().tolist()

    return grid_map

  @property
  def map(self) -> OccupancyGrid:
    """Read-only access to the loaded occupancy grid."""
    return self._map

  @map.setter
  def map(self, grid_map: OccupancyGrid) -> None:
    self._map = grid_map

  def is_in_bounds(self, gx, gy) -> bool:
    return 0 <= gx < self.map.width and 0 <= gy < self.map.height

  def get_probability(self, gx: int, gy: int) -> Optional[int]:
    if not self.is_in_bounds(gx, gy):
      return None
    return self.map.data[gy + gx * self.map.width]
