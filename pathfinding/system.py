"""pairwise-pathfinding-system - PathfindingSystem.

Grid based pathfinding with:
* find_path: 8-direction A* using an octile-distance heuristic, a binary-heap
  open list and line-of-sight path smoothing.
* check_obstacle: full agent-volume (radius) collision tests against static
  and dynamic obstacles, cached and invalidated on world changes.
* build_navigation_mesh: connectivity-checked, contour-simplified,
  hierarchical navigation mesh.
* request_path: asynchronous, FIFO-queued, cached path requests.
* follow_path: waypoint arrival detection, blocking re-pathing and
  turn-aware speed control.

Coordinates are (x, y); grids are sequences of rows, grid[y][x] == 1 blocked.
"""

from __future__ import annotations

import heapq
import math
import queue
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

Point = Tuple[float, float]
Grid = Sequence[Sequence[int]]

SQRT2_MINUS_1 = math.sqrt(2.0) - 1.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _grid_dims(grid: Grid) -> Tuple[int, int]:
    return (len(grid[0]) if grid else 0, len(grid))


def _octile(a: Point, b: Point) -> float:
    """Admissible heuristic for 8-direction movement.

    Manhattan distance overestimates diagonal moves and is inadmissible on a
    diagonal grid, which makes A* take detours; octile distance is exact on
    an open grid, so A* stays optimal.
    """
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + SQRT2_MINUS_1 * min(dx, dy)


def _euclid(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _supercover(x0: float, y0: float, x1: float, y1: float) -> List[Tuple[int, int]]:
    """All grid cells touched by segment (x0,y0)->(x1,y1) (Amanatides-Woo)."""
    cells: List[Tuple[int, int]] = []
    ix, iy = int(math.floor(x0)), int(math.floor(y0))
    end_x, end_y = int(math.floor(x1)), int(math.floor(y1))
    dx, dy = x1 - x0, y1 - y0
    step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
    step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)

    if dx == 0.0:
        t_max_x = math.inf
        t_delta_x = math.inf
    else:
        boundary = ix + 1 if step_x > 0 else ix
        t_max_x = (boundary - x0) / dx
        t_delta_x = abs(1.0 / dx)
    if dy == 0.0:
        t_max_y = math.inf
        t_delta_y = math.inf
    else:
        boundary = iy + 1 if step_y > 0 else iy
        t_max_y = (boundary - y0) / dy
        t_delta_y = abs(1.0 / dy)

    cells.append((ix, iy))
    while (ix, iy) != (end_x, end_y):
        if math.isclose(t_max_x, t_max_y, rel_tol=0.0, abs_tol=1e-9):
            # Corner crossing: include both orthogonal neighbours so a
            # diagonal squeeze between two touching obstacles is rejected.
            cells.append((ix + step_x, iy))
            cells.append((ix, iy + step_y))
            ix += step_x
            iy += step_y
            t_max_x += t_delta_x
            t_max_y += t_delta_y
        elif t_max_x < t_max_y:
            ix += step_x
            t_max_x += t_delta_x
        else:
            iy += step_y
            t_max_y += t_delta_y
        cells.append((ix, iy))
    return cells


def _line_of_sight(a: Point, b: Point, blocked: Callable[[int, int], bool]) -> bool:
    """True when the swept segment between cell centers touches no obstacle."""
    for cx, cy in _supercover(a[0] + 0.5, a[1] + 0.5, b[0] + 0.5, b[1] + 0.5):
        if blocked(cx, cy):
            return False
    return True


def _simplify_polyline(points: List[Point],
                       blocked: Callable[[int, int], bool]) -> List[Point]:
    """String-pulling smoothing: drop waypoints while line-of-sight survives."""
    if len(points) <= 2:
        return list(points)
    smoothed = [points[0]]
    anchor = points[0]
    i = 1
    while i < len(points):
        if _line_of_sight(anchor, points[i], blocked):
            i += 1
            continue
        smoothed.append(points[i - 1])
        anchor = points[i - 1]
    if smoothed[-1] != points[-1]:
        smoothed.append(points[-1])
    return smoothed


def _point_to_cell_distance(px: float, py: float, cx: int, cy: int) -> float:
    """Shortest distance from point (px, py) to grid cell (cx, cy)'s AABB."""
    dx = max(float(cx) - px, 0.0, px - (cx + 1))
    dy = max(float(cy) - py, 0.0, py - (cy + 1))
    return math.hypot(dx, dy)


# ---------------------------------------------------------------------------
# Navigation mesh
# ---------------------------------------------------------------------------

@dataclass
class NavMesh:
    """Connectivity-checked, simplified, hierarchical navigation mesh."""

    grid: List[List[int]]
    width: int
    height: int
    regions: List[List[Tuple[int, int]]]
    region_ids: List[List[int]]
    contours: List[List[Tuple[float, float]]]
    simplified_contours: List[List[Tuple[float, float]]]
    layers: List[List[List[int]]]
    connected: bool

    def is_connected(self, a: Point, b: Point) -> bool:
        ra = self.region_ids[int(a[1])][int(a[0])]
        rb = self.region_ids[int(b[1])][int(b[0])]
        return ra >= 0 and ra == rb

    def layer_count(self) -> int:
        return len(self.layers)


# ---------------------------------------------------------------------------
# Requests and following
# ---------------------------------------------------------------------------

@dataclass
class PathRequest:
    """Handle for an asynchronous path request."""

    request_id: int
    start: Point
    goal: Point
    _event: threading.Event = field(default_factory=threading.Event, repr=False)
    _path: Optional[List[Point]] = None
    _error: Optional[BaseException] = None

    def _resolve(self, path: Optional[List[Point]]) -> None:
        self._path = path
        self._event.set()

    @property
    def done(self) -> bool:
        return self._event.is_set()

    @property
    def path(self) -> Optional[List[Point]]:
        return self._path if self.done else None

    def wait(self, timeout: Optional[float] = None) -> Optional[List[Point]]:
        self._event.wait(timeout)
        return self._path

    def get(self, timeout: Optional[float] = None) -> Optional[List[Point]]:
        if not self._event.wait(timeout):
            raise TimeoutError("path request did not complete in time")
        if self._error is not None:
            raise self._error
        return self._path


@dataclass
class FollowResult:
    position: Point
    velocity: Point
    speed: float
    target_index: int
    arrived: bool
    repathed: bool
    path: List[Point]


class PathfindingSystem:
    """Grid pathfinding, navigation mesh, async requests and path following."""

    DEFAULT_GRID: List[List[int]] = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    def __init__(self, grid: Optional[Grid] = None):
        self.grid: List[List[int]] = [list(row) for row in (grid or self.DEFAULT_GRID)]
        self.width, self.height = _grid_dims(self.grid)
        self._grid_version = 0

        # Dynamic obstacles: obstacle_id -> (x, y, radius)
        self.dynamic_obstacles: Dict[Any, Tuple[float, float, float]] = {}
        self._dynamic_version = 0

        # Cached volume collision checks: key -> (version, result)
        self._obstacle_cache: Dict[Any, Tuple[Tuple[int, int], bool]] = {}
        self._cache_lock = threading.Lock()

        # Async request machinery: FIFO queue, worker thread, result cache.
        self._request_queue: "queue.Queue[Optional[PathRequest]]" = queue.Queue()
        self._request_counter = 0
        self._request_cache: Dict[Any, Optional[List[Point]]] = {}
        self._inflight: Dict[Any, List[PathRequest]] = {}
        self._request_lock = threading.Lock()
        self._worker = threading.Thread(target=self._process_requests,
                                        name="pathfinding-worker", daemon=True)
        self._worker.start()

        self._nav_mesh: Optional[NavMesh] = None

    # ------------------------------------------------------------------
    # Obstacle handling
    # ------------------------------------------------------------------

    def add_dynamic_obstacle(self, obstacle_id: Any, x: float, y: float,
                             radius: float = 0.5) -> None:
        """Register or move a dynamic obstacle; bumps the cache version."""
        self.dynamic_obstacles[obstacle_id] = (float(x), float(y), float(radius))
        self._dynamic_version += 1

    def remove_dynamic_obstacle(self, obstacle_id: Any) -> None:
        self.dynamic_obstacles.pop(obstacle_id, None)
        self._dynamic_version += 1

    def update_dynamic_obstacle(self, obstacle_id: Any, x: float, y: float,
                                radius: Optional[float] = None) -> None:
        """Move a dynamic obstacle and refresh the navigation representation."""
        old_radius = self.dynamic_obstacles.get(obstacle_id, (x, y, radius or 0.5))[2]
        self.add_dynamic_obstacle(obstacle_id, x, y,
                                  old_radius if radius is None else radius)

    def _static_blocked(self, cx: int, cy: int) -> bool:
        if not (0 <= cx < self.width and 0 <= cy < self.height):
            return True
        return self.grid[cy][cx] == 1

    def _dynamic_blocked(self, cx: int, cy: int) -> bool:
        for ox, oy, radius in self.dynamic_obstacles.values():
            if math.hypot(cx + 0.5 - ox, cy + 0.5 - oy) <= radius:
                return True
        return False

    def _volume_blocked(self, px: float, py: float, radius: float) -> bool:
        """True if the whole agent disk centered at (px, py) collides.

        Static cells are tested with exact circle-vs-cell distance, so an
        off-center agent whose disk crosses into a wall is detected; out of
        bounds cells count as solid. Touching a wall exactly (distance ==
        radius) is allowed.
        """
        px, py = float(px), float(py)
        cx, cy = int(math.floor(px)), int(math.floor(py))
        if radius <= 1e-9:
            if self._static_blocked(cx, cy):
                return True
        else:
            limit = radius - 1e-9
            lo_x, hi_x = int(math.floor(px - radius)), int(math.floor(px + radius))
            lo_y, hi_y = int(math.floor(py - radius)), int(math.floor(py + radius))
            for ny in range(lo_y, hi_y + 1):
                for nx in range(lo_x, hi_x + 1):
                    in_bounds = 0 <= nx < self.width and 0 <= ny < self.height
                    solid = (not in_bounds) or self.grid[ny][nx] == 1
                    if solid and _point_to_cell_distance(px, py, nx, ny) < limit:
                        return True
        for ox, oy, obs_r in self.dynamic_obstacles.values():
            if math.hypot(px - ox, py - oy) <= radius + obs_r:
                return True
        return False

    @staticmethod
    def _disk_hits_grid(px: float, py: float, radius: float,
                        grid: Optional[Grid]) -> bool:
        """Exact circle-vs-grid volume test for an externally supplied grid."""
        width, height = _grid_dims(grid)
        cx, cy = int(math.floor(px)), int(math.floor(py))
        if radius <= 1e-9:
            return not (0 <= cx < width and 0 <= cy < height) or grid[cy][cx] == 1
        limit = radius - 1e-9
        lo_x, hi_x = int(math.floor(px - radius)), int(math.floor(px + radius))
        lo_y, hi_y = int(math.floor(py - radius)), int(math.floor(py + radius))
        for ny in range(lo_y, hi_y + 1):
            for nx in range(lo_x, hi_x + 1):
                if not (0 <= nx < width and 0 <= ny < height):
                    solid, in_bounds = True, False
                else:
                    solid, in_bounds = grid[ny][nx] == 1, True
                if solid and (not in_bounds
                              or _point_to_cell_distance(px, py, nx, ny) < limit):
                    return True
        return False

    def check_obstacle(self, position: Optional[Point] = None,
                       radius: float = 0.4,
                       grid: Optional[Grid] = None,
                       dynamic_obstacles: Optional[Dict[Any, Tuple[float, float, float]]] = None,
                       use_cache: bool = True,
                       **kwargs: Any) -> bool:
        """Volume based collision test.

        Every cell overlapped by the agent disk (not just the center cell) is
        tested against static and dynamic obstacles. Results are cached per
        position/radius and invalidated automatically when the world changes.
        Returns True when the position collides with an obstacle.
        """
        if position is None:
            position, radius = (0.5, 0.5), 0.4
        radius = float(radius)

        if grid is None and dynamic_obstacles is None and use_cache:
            key = (round(float(position[0]), 4), round(float(position[1]), 4),
                   round(radius, 4))
            version = (self._grid_version, self._dynamic_version)
            with self._cache_lock:
                cached = self._obstacle_cache.get(key)
                if cached is not None and cached[0] == version:
                    return cached[1]
            result = self._volume_blocked(position[0], position[1], radius)
            with self._cache_lock:
                self._obstacle_cache[key] = (version, result)
            return result

        # Uncached path for explicit grids / obstacle sets.
        px, py = float(position[0]), float(position[1])
        if grid is not None:
            if self._disk_hits_grid(px, py, radius, grid):
                return True
        elif self._volume_blocked(px, py, radius):
            return True
        obstacles = dynamic_obstacles if dynamic_obstacles is not None else self.dynamic_obstacles
        for ox, oy, obs_r in obstacles.values():
            if math.hypot(px - ox, py - oy) <= radius + obs_r:
                return True
        return False

    # ------------------------------------------------------------------
    # A* pathfinding
    # ------------------------------------------------------------------

    def _blocked_predicate(self, radius: float,
                           ignore: Iterable[Point] = ()) -> Callable[[int, int], bool]:
        ignored = set((int(p[0]), int(p[1])) for p in ignore)

        def blocked(cx: int, cy: int) -> bool:
            if (cx, cy) in ignored:
                return False
            return self._volume_blocked(cx + 0.5, cy + 0.5, radius)

        return blocked

    def find_path(self, start: Optional[Point] = None, goal: Optional[Point] = None,
                  radius: float = 0.0, grid: Optional[Grid] = None,
                  smooth: bool = True, diagonal: bool = True,
                  dynamic_obstacles: Optional[Dict[Any, Tuple[float, float, float]]] = None,
                  **kwargs: Any) -> Optional[List[Point]]:
        """8-direction A* with octile heuristic, heap open list and smoothing.

        Returns the list of cell coordinates from ``start`` to ``goal`` or
        ``None`` when no path exists.
        """
        system = self
        if grid is not None or dynamic_obstacles is not None:
            system = PathfindingSystem(grid)
            if dynamic_obstacles:
                system.dynamic_obstacles = dict(dynamic_obstacles)
        if start is None:
            start, goal = (0, 0), (system.width - 1, system.height - 1)
        if goal is None:
            goal = (system.width - 1, system.height - 1)

        sx, sy = int(round(start[0])), int(round(start[1]))
        gx, gy = int(round(goal[0])), int(round(goal[1]))
        if not (0 <= sx < system.width and 0 <= sy < system.height
                and 0 <= gx < system.width and 0 <= gy < system.height):
            return None

        # Start/goal themselves are trusted even if they touch an inflated wall.
        blocked = system._blocked_predicate(radius, ignore=((sx, sy), (gx, gy)))
        if (sx, sy) == (gx, gy):
            return [(sx, sy)]

        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        if diagonal:
            directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        directions = [(dx, dy, math.sqrt(2.0) if dx and dy else 1.0)
                      for dx, dy in directions]

        goal_cell = (gx, gy)
        open_heap: List[Tuple[float, int, Tuple[int, int]]] = []
        counter = 0
        heapq.heappush(open_heap, (_octile((sx, sy), goal_cell), counter, (sx, sy)))
        g_score: Dict[Tuple[int, int], float] = {(sx, sy): 0.0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        closed: set = set()

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal_cell:
                path = self._reconstruct(came_from, current)
                if smooth:
                    path = _simplify_polyline(path, blocked)
                return path
            closed.add(current)

            cx, cy = current
            for dx, dy, step_cost in directions:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < system.width and 0 <= ny < system.height):
                    continue
                neighbour = (nx, ny)
                if neighbour in closed or blocked(nx, ny):
                    continue
                if dx and dy and (blocked(cx + dx, cy) or blocked(cx, cy + dy)):
                    continue  # No corner cutting through diagonal squeezes.
                tentative = g_score[current] + step_cost
                if tentative < g_score.get(neighbour, math.inf):
                    came_from[neighbour] = current
                    g_score[neighbour] = tentative
                    f_value = tentative + _octile(neighbour, goal_cell)
                    counter += 1
                    heapq.heappush(open_heap, (f_value, counter, neighbour))
        return None

    @staticmethod
    def _reconstruct(came_from: Dict[Tuple[int, int], Tuple[int, int]],
                     end: Tuple[int, int]) -> List[Point]:
        path = [end]
        while end in came_from:
            end = came_from[end]
            path.append(end)
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # Navigation mesh
    # ------------------------------------------------------------------

    def build_navigation_mesh(self, grid: Optional[Grid] = None,
                              levels: int = 3, **kwargs: Any) -> NavMesh:
        """Build a connectivity-checked, simplified, hierarchical nav mesh.

        * Connectivity: walkable cells are flood-filled into regions; cells
          that cannot reach the main region are reported, not silently kept.
        * Simplification: region contours are traced and collinear vertices
          are removed so obstacle edges do not explode the vertex count.
        * Hierarchy: progressively coarser layers (2x, 4x, ...) let large maps
          plan at coarse level first instead of one giant fine grid.
        """
        source = [list(row) for row in (grid if grid is not None else self.grid)]
        width, height = _grid_dims(source)

        region_ids = [[-1] * width for _ in range(height)]
        regions: List[List[Tuple[int, int]]] = []
        for y in range(height):
            for x in range(width):
                if source[y][x] == 1 or region_ids[y][x] != -1:
                    continue
                region: List[Tuple[int, int]] = []
                frontier = deque([(x, y)])
                region_ids[y][x] = len(regions)
                while frontier:
                    cx, cy = frontier.popleft()
                    region.append((cx, cy))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = cx + dx, cy + dy
                        if (0 <= nx < width and 0 <= ny < height
                                and source[ny][nx] == 0
                                and region_ids[ny][nx] == -1):
                            region_ids[ny][nx] = len(regions)
                            frontier.append((nx, ny))
                regions.append(region)

        contours = [self._trace_contour(source, region) for region in regions]
        simplified = [self._simplify_contour(contour) for contour in contours]
        layers = self._build_layers(source, levels)

        mesh = NavMesh(grid=source, width=width, height=height,
                       regions=regions, region_ids=region_ids,
                       contours=contours, simplified_contours=simplified,
                       layers=layers, connected=len(regions) <= 1)
        if grid is None:
            self._nav_mesh = mesh
        return mesh

    @staticmethod
    def _trace_contour(grid: List[List[int]],
                       region: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        """Trace the boundary of a walkable region as a vertex loop."""
        cells = set(region)
        width, height = _grid_dims(grid)
        edges: Dict[Tuple[float, float], List[Tuple[float, float]]] = {}

        def add_edge(a: Tuple[float, float], b: Tuple[float, float]) -> None:
            edges.setdefault(a, []).append(b)

        for (cx, cy) in cells:
            x0, y0, x1, y1 = float(cx), float(cy), float(cx + 1), float(cy + 1)
            if (cx, cy - 1) not in cells:  # top edge
                add_edge((x0, y0), (x1, y0))
            if (cx + 1, cy) not in cells:  # right edge
                add_edge((x1, y0), (x1, y1))
            if (cx, cy + 1) not in cells:  # bottom edge
                add_edge((x1, y1), (x0, y1))
            if (cx - 1, cy) not in cells:  # left edge
                add_edge((x0, y1), (x0, y0))

        if not edges:
            return []
        contour = []
        start = min(edges)
        current = start
        visited_edges = 0
        max_edges = sum(len(v) for v in edges.values()) + 1
        while visited_edges < max_edges:
            contour.append(current)
            neighbours = edges.get(current)
            if not neighbours:
                break
            current = neighbours.pop()
            visited_edges += 1
            if current == start:
                break
        return contour

    @staticmethod
    def _simplify_contour(contour: List[Tuple[float, float]],
                          tolerance: float = 1e-9) -> List[Tuple[float, float]]:
        """Drop collinear vertices so obstacle edges keep only corners."""
        if len(contour) < 3:
            return list(contour)
        simplified: List[Tuple[float, float]] = []
        count = len(contour)
        for i, point in enumerate(contour):
            prev = contour[(i - 1) % count]
            nxt = contour[(i + 1) % count]
            cross = ((point[0] - prev[0]) * (nxt[1] - point[1])
                     - (point[1] - prev[1]) * (nxt[0] - point[0]))
            if abs(cross) > tolerance:
                simplified.append(point)
        return simplified or list(contour[:1])

    @staticmethod
    def _build_layers(grid: List[List[int]], levels: int) -> List[List[List[int]]]:
        """Hierarchical layers: each level halves resolution (OR-combined)."""
        layers = [grid]
        current = grid
        for _ in range(1, max(1, levels)):
            width, height = _grid_dims(current)
            coarse: List[List[int]] = []
            for y in range(0, height, 2):
                row = []
                for x in range(0, width, 2):
                    blocked = 0
                    for oy in (0, 1):
                        for ox in (0, 1):
                            if (y + oy < height and x + ox < width
                                    and current[y + oy][x + ox] == 1):
                                blocked = 1
                    row.append(blocked)
                coarse.append(row)
            if coarse == current:
                break
            layers.append(coarse)
            current = coarse
        return layers

    # ------------------------------------------------------------------
    # Asynchronous path requests
    # ------------------------------------------------------------------

    def request_path(self, start: Optional[Point] = None,
                     goal: Optional[Point] = None,
                     radius: float = 0.0,
                     wait: bool = False,
                     timeout: Optional[float] = None,
                     **kwargs: Any) -> Any:
        """Queue an asynchronous path request.

        Requests are processed in FIFO order by a background worker, so the
        calling thread never blocks on the search itself. Identical
        (start, goal, radius) requests share one cached result; duplicate
        in-flight requests are attached to the same computation. Returns a
        :class:`PathRequest` handle, or the path directly when ``wait=True``.
        """
        if start is None:
            start, goal = (0, 0), (self.width - 1, self.height - 1)
        if goal is None:
            goal = (self.width - 1, self.height - 1)
        key = (tuple(start), tuple(goal), float(radius),
               self._grid_version, self._dynamic_version)

        with self._request_lock:
            if key in self._request_cache:
                cached = self._request_cache[key]
                if wait:
                    return cached
                request = self._new_request(start, goal)
                request._resolve(cached)
                return request
            waiters = self._inflight.get(key)
            if waiters is None:
                waiters = []
                self._inflight[key] = waiters
                enqueue = True
            else:
                enqueue = False
            request = self._new_request(start, goal)
            waiters.append(request)

        if enqueue:
            self._request_queue.put((key, start, goal, radius))

        if wait:
            return request.get(timeout)
        return request

    def _new_request(self, start: Point, goal: Point) -> PathRequest:
        self._request_counter += 1
        return PathRequest(request_id=self._request_counter,
                           start=tuple(start), goal=tuple(goal))

    def _process_requests(self) -> None:
        """Worker loop: strictly FIFO so queued requests never reorder."""
        while True:
            item = self._request_queue.get()
            if item is None:
                return
            key, start, goal, radius = item
            try:
                path = self.find_path(start, goal, radius=radius)
            except BaseException as exc:  # pragma: no cover - defensive
                path = None
                error = exc
            else:
                error = None
            with self._request_lock:
                self._request_cache[key] = path
                waiters = self._inflight.pop(key, [])
            for request in waiters:
                request._error = error
                request._resolve(path)
            self._request_queue.task_done()

    def shutdown(self) -> None:
        self._request_queue.put(None)

    # ------------------------------------------------------------------
    # Path following
    # ------------------------------------------------------------------

    def follow_path(self, position: Optional[Point] = None,
                    path: Optional[Sequence[Point]] = None,
                    speed: float = 1.0, dt: float = 1.0,
                    arrive_distance: float = 0.5,
                    turn_slowdown: float = 0.5,
                    radius: float = 0.0,
                    goal: Optional[Point] = None,
                    **kwargs: Any) -> FollowResult:
        """Advance one step along ``path``.

        * Arrival detection: waypoints within ``arrive_distance`` are consumed
          instead of being pushed against forever.
        * Re-pathing: if the next waypoint is blocked (e.g. a dynamic
          obstacle moved in), a fresh path to the goal is computed.
        * Speed control: sharp turns scale the speed down by
          ``turn_slowdown`` so corners are not taken at full velocity.
        """
        if position is None:
            position = (0.5, 0.5)
        if path is None:
            if goal is None:
                goal = (self.width - 1, self.height - 1)
            path = self.find_path((int(position[0]), int(position[1])), goal,
                                  radius=radius) or []
        waypoints = [tuple(p) for p in path]
        repathed = False

        # Arrival detection: snap to the nearest waypoint, then consume every
        # waypoint within arrive_distance instead of pushing past it.
        index = (min(range(len(waypoints)),
                     key=lambda i: _euclid(position, waypoints[i]))
                 if waypoints else 0)
        while (index < len(waypoints) - 1
               and _euclid(position, waypoints[index]) <= arrive_distance):
            index += 1
        if not waypoints or _euclid(position, waypoints[-1]) <= arrive_distance:
            return FollowResult(position=position, velocity=(0.0, 0.0), speed=0.0,
                                target_index=max(len(waypoints) - 1, 0),
                                arrived=True, repathed=False, path=waypoints)

        target = waypoints[index]

        # Re-path when the next waypoint is blocked.
        if self._volume_blocked(int(target[0]) + 0.5, int(target[1]) + 0.5,
                                max(radius, 0.4)):
            goal_cell = waypoints[-1]
            new_path = self.find_path((int(position[0]), int(position[1])),
                                      goal_cell, radius=radius)
            if new_path:
                waypoints = new_path
                index = 0
                target = waypoints[0]
                repathed = True
            else:
                return FollowResult(position=position, velocity=(0.0, 0.0),
                                    speed=0.0, target_index=index,
                                    arrived=False, repathed=True, path=waypoints)

        # Turn-aware speed control.
        direction = (target[0] - position[0], target[1] - position[1])
        distance = math.hypot(*direction)
        effective_speed = float(speed)
        if index + 1 < len(waypoints) and distance > 1e-9:
            nxt = waypoints[index + 1]
            out_dir = (nxt[0] - target[0], nxt[1] - target[1])
            out_len = math.hypot(*out_dir)
            if out_len > 1e-9:
                cos_turn = ((direction[0] * out_dir[0] + direction[1] * out_dir[1])
                            / (distance * out_len))
                turn_factor = (1.0 + cos_turn) / 2.0  # 1 straight, 0 reversal
                effective_speed = speed * (turn_slowdown
                                           + (1.0 - turn_slowdown) * turn_factor)

        step = min(effective_speed * dt, distance)
        velocity = (direction[0] / distance * step, direction[1] / distance * step)
        new_position = (position[0] + velocity[0], position[1] + velocity[1])
        arrived = _euclid(new_position, waypoints[-1]) <= arrive_distance
        return FollowResult(position=new_position, velocity=velocity,
                            speed=step / dt if dt > 0 else 0.0,
                            target_index=index, arrived=arrived,
                            repathed=repathed, path=waypoints)
