from dataclasses import dataclass, field
import json
from typing import List, Optional, Tuple

import math

@dataclass
class Pose2D:
  x: float = 0.0
  y: float = 0.0
  theta: float = 0.0

@dataclass
class OccupancyGrid:
  resolution: float = 0.05
  width: int = 30
  height: int = 30
  origin: Pose2D = field(default_factory=Pose2D)
  data: List[int] = field(default_factory=list)

  def map_to_grid(self, mx: float, my: float) -> tuple[int, int]:
    gx = math.floor((mx - self.origin.x) / self.resolution)
    gy = math.floor((my - self.origin.y) / self.resolution)

    return gx, gy

  def grid_to_map(self, gx: int, gy: int) -> tuple[int, int]:
    mx = self.origin.x + (gx + 0.5) * self.resolution
    my = self.origin.y + (gy + 0.5) * self.resolution

    return mx, my

  def get_cost(self, gx: int, gy: int) -> int:
    return self.data[self.get_grid_index(gx, gy)]

  def get_grid_index(self, gx, gy):
    return gx + gy * self.width

@dataclass
class Node2D:
  index: int = -1
  x: int = 0
  y: int = 0
  g: float = 0.0
  h: float = 0.0
  f: float = 0.0
  parent: Optional["Node2D"] = None

  def __eq__(self, n: "Node2D"):
    return self.x == n.x and self.y == n.y

  def cost(self, goal: "Node2D"):
    self.h = math.hypot(self.x - goal.x, self.y - goal.y)
    self.f = self.h + self.g

@dataclass
class HybridNode:
  x: float
  y: float
  yaw: float
  g: float
  h: float = 0.0
  f: float = 0.0
  parent: Optional["HybridNode"] = None

  def cost(self, goal: Tuple[float, float]) -> None:
    self.h = math.hypot(self.x - goal[0], self.y - goal[1])
    self.f = self.g + self.h


@dataclass
class HybridAStarParams:
  max_steer: float = 0.5
  wheel_base: float = 3.0
  dt: float = 0.2
  v: float = 0.5
  goal_xy_tolerance: float = 0.5
  goal_yaw_tolerance: float = math.radians(10.0)
  yaw_discretization: int = 72

@dataclass
class FootPrint:
  def __init__(self, config):
    with open(config, "r") as f:
      footprint_config = json.load(f)

    self._width = footprint_config["width"]
    self._length = footprint_config["length"]

    self._footprint = [
      [self._length / 2.0, self._width / 2.0],
      [self._length / 2.0, -self._width / 2.0],
      [-self._length / 2.0, -self._width / 2.0],
      [-self._length / 2.0, self._width / 2.0],
    ]

  @property
  def footprint(self):
    return self._footprint

  @property
  def width(self) -> float:
    return self._width

  @property
  def length(self) -> float:
    return self._length
