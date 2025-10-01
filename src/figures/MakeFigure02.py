#!/usr/bin/env python3
"""Representative pattern figure generator."""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

# Ensure repo root on sys.path for direct execution (mirrors other figure scripts)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.plotting.plotting import DEFAULT_FIGURE_DPI
from src.utils.plotting.figure_naming import figure_output_dir, figure_output_stem


CARPET_COLORS: Tuple[str, ...] = (
    "#ff4444",
    "#ff8800",
    "#ffdd00",
    "#88ff00",
    "#00ddff",
    "#8844ff",
    "#ff44ff",
    "#ffffff",
)

EMPTY_COLOR = "#f0f0f0"

SPATIAL_SAMPLES = 3000
SPATIAL_ATTEMPTS_MAX = 15000
COLOR_SAMPLES = 3000
SCATTER_COLOR_SAMPLES = 61
JOINT_SPATIAL_SAMPLES = 600
JOINT_COLOR_ASSIGNMENTS = 15

EIGHT_NEIGHBORS: Tuple[Tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),             (0, 1),
    (1, -1),  (1, 0),   (1, 1),
)

DEFAULT_GRID_SIZE = 5
DEFAULT_PAIRS: Tuple[Tuple[int, int], ...] = ((7, 2), (10, 3))
DEFAULT_PERCENTILES: Tuple[int, ...] = (1, 50, 99)


class PatternGrid:
    """Grid helper mirroring the JS PatternGrid behaviour."""

    def __init__(self, grid_size: int) -> None:
        self.grid_size = grid_size
        self.grid: List[List[int]] = [[0 for _ in range(grid_size)] for _ in range(grid_size)]

    def set_cell(self, row: int, col: int, value: int) -> None:
        if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
            self.grid[row][col] = value

    def get_cell(self, row: int, col: int) -> int:
        if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
            return self.grid[row][col]
        return 0

    def get_occupied_positions(self) -> List[Tuple[int, int]]:
        positions: List[Tuple[int, int]] = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r][c] > 0:
                    positions.append((r, c))
        return positions

    def copy(self) -> "PatternGrid":
        new_grid = PatternGrid(self.grid_size)
        for r in range(self.grid_size):
            new_grid.grid[r][:] = self.grid[r][:]
        return new_grid


@dataclass
class PatternMetrics:
    pattern: PatternGrid
    spatial_entropy: float
    edges: int
    max_possible_edges: int
    is_connected: bool
    generation_time: float
    achieved_spatial_percentile: float
    achieved_color_percentile: float
    max_color_components: int
    block_count: int
    color_count: int
    spatial_target: float
    color_target: float


@dataclass
class ScatterPoint:
    entropy: float
    edges: float
    label: str
    is_random: bool
    percentile: Optional[float] = None
    spatial_percentile: Optional[float] = None
    color_percentile: Optional[float] = None
    color_index: Optional[int] = None
    pair_index: Optional[int] = None
    max_edges: Optional[float] = None


class PatternGenerator:
    """Port of the browser pattern generator with identical logic."""

    def __init__(
        self,
        grid_size: int = DEFAULT_GRID_SIZE,
        *,
        use_fixed_seed: bool = False,
        seed_value: int = 12345,
        random_sample_count: int = 10_000,
        percentile_mode: str = "conditional",
    ) -> None:
        self.grid_size = grid_size
        self.use_fixed_seed = use_fixed_seed
        self.seed_value = seed_value & 0xFFFFFFFF
        self.lcg_state = self.seed_value
        self.random_sample_count = random_sample_count
        self.percentile_mode = percentile_mode
        self.joint_dataset_cache: Dict[str, List[Dict[str, object]]] = {}

    @staticmethod
    def _now() -> float:
        return time.perf_counter()

    # --- Random helpers -------------------------------------------------

    def _set_seed(self, value: int) -> None:
        self.lcg_state = value & 0xFFFFFFFF

    def _prepare_seed(self, offset: int = 0) -> None:
        if self.use_fixed_seed:
            self._set_seed(self.seed_value + offset)

    def _random(self) -> float:
        if not self.use_fixed_seed:
            return random.random()
        self.lcg_state = (self.lcg_state * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.lcg_state / 0x100000000

    def _random_int(self, max_value: int) -> int:
        if max_value <= 0:
            return 0
        return int(math.floor(self._random() * max_value))

    # --- Percentile helpers ---------------------------------------------

    @staticmethod
    def _get_percentile_selection(length: int, percentile: float) -> Tuple[int, float]:
        if length <= 0:
            return 0, 0.0
        capped = max(0.0, min(100.0, float(percentile)))
        if length == 1:
            return 0, 0.0
        if capped <= 1.0:
            return 0, 0.0
        if capped >= 99.0:
            return length - 1, 100.0
        normalized = capped / 100.0
        index = max(0, min(length - 1, round(normalized * (length - 1))))
        achieved = (index / (length - 1)) * 100.0 if length > 1 else 0.0
        return index, achieved

    # --- Geometry + metrics --------------------------------------------

    def _has_symmetry(self, pattern: PatternGrid) -> bool:
        n = self.grid_size

        def occupied(r: int, c: int) -> bool:
            return pattern.get_cell(r, c) > 0

        def equal_to(transform: callable) -> bool:
            for r in range(n):
                for c in range(n):
                    if occupied(r, c) != transform(r, c):
                        return False
            return True

        mirror_h = lambda r, c: occupied(r, n - 1 - c)
        mirror_v = lambda r, c: occupied(n - 1 - r, c)
        diag_tlbr = lambda r, c: occupied(c, r)
        diag_trbl = lambda r, c: occupied(n - 1 - c, n - 1 - r)

        return any(
            equal_to(fn)
            for fn in (mirror_h, mirror_v, diag_tlbr, diag_trbl)
        )

    def _is_connected(self, pattern: PatternGrid) -> bool:
        positions = pattern.get_occupied_positions()
        if not positions:
            return True
        visited: set[str] = set()
        queue: deque[Tuple[int, int]] = deque([positions[0]])
        visited.add(f"{positions[0][0]},{positions[0][1]}")

        while queue:
            r, c = queue.popleft()
            for dr, dc in EIGHT_NEIGHBORS:
                nr, nc = r + dr, c + dc
                key = f"{nr},{nc}"
                if pattern.get_cell(nr, nc) > 0 and key not in visited:
                    visited.add(key)
                    queue.append((nr, nc))

        return len(visited) == len(positions)

    def _calculate_spatial_entropy(self, pattern: PatternGrid) -> float:
        """Blend dispersion (pairwise distance) with branching (clustering coefficient)."""

        positions = pattern.get_occupied_positions()
        count = len(positions)
        if count <= 1:
            return 0.0

        # Pairwise Manhattan distance component
        total_distance = 0.0
        pair_count = 0
        for idx, (r1, c1) in enumerate(positions):
            for r2, c2 in positions[idx + 1 :]:
                total_distance += abs(r1 - r2) + abs(c1 - c2)
                pair_count += 1

        dispersion = 0.0
        if pair_count > 0:
            mean_distance = total_distance / pair_count
            max_distance = 2 * (self.grid_size - 1)
            if max_distance > 0:
                dispersion = max(0.0, min(1.0, mean_distance / float(max_distance)))

        # Local clustering coefficient across the 8-neighbour graph
        occupied_set = {(r, c) for r, c in positions}
        clustering_values: List[float] = []
        for r, c in positions:
            neighbours = []
            for dr, dc in EIGHT_NEIGHBORS:
                nr, nc = r + dr, c + dc
                if (nr, nc) in occupied_set:
                    neighbours.append((nr, nc))
            k = len(neighbours)
            if k < 2:
                continue
            links = 0
            possible = k * (k - 1) / 2
            for i, (nr1, nc1) in enumerate(neighbours):
                for nr2, nc2 in neighbours[i + 1 :]:
                    if max(abs(nr1 - nr2), abs(nc1 - nc2)) <= 1:
                        links += 1
            if possible > 0:
                clustering_values.append(links / possible)

        if clustering_values:
            clustering = sum(clustering_values) / len(clustering_values)
        else:
            clustering = 0.0

        # Favour dispersed-but-branching patterns: high dispersion + low clustering
        combined = 0.6 * dispersion + 0.4 * (1.0 - clustering)
        return max(0.0, min(1.0, combined))

    def _generate_connected_spatial_pattern(self, block_count: int) -> PatternGrid:
        pattern = PatternGrid(self.grid_size)
        start_r = self._random_int(self.grid_size)
        start_c = self._random_int(self.grid_size)
        pattern.set_cell(start_r, start_c, 1)

        occupied: set[str] = {f"{start_r},{start_c}"}
        frontier: List[Tuple[int, int]] = []

        for dr, dc in EIGHT_NEIGHBORS:
            nr, nc = start_r + dr, start_c + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                key = f"{nr},{nc}"
                if key not in occupied:
                    frontier.append((nr, nc))

        while len(occupied) < block_count and frontier:
            idx = self._random_int(len(frontier))
            r, c = frontier.pop(idx)
            key = f"{r},{c}"
            if key in occupied:
                continue
            pattern.set_cell(r, c, 1)
            occupied.add(key)

            for dr, dc in EIGHT_NEIGHBORS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    neighbor_key = f"{nr},{nc}"
                    if neighbor_key not in occupied and not any(fr == nr and fc == nc for fr, fc in frontier):
                        frontier.append((nr, nc))

        return pattern

    def _count_occupied_adjacencies(self, pattern: PatternGrid) -> int:
        edge_count = 0
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if pattern.get_cell(r, c) == 0:
                    continue
                for dr, dc in EIGHT_NEIGHBORS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        if pattern.get_cell(nr, nc) > 0:
                            edge_count += 1
        return edge_count // 2

    def _count_different_color_edges(self, pattern: PatternGrid) -> int:
        edge_count = 0
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                color1 = pattern.get_cell(r, c)
                if color1 == 0:
                    continue
                for dr, dc in EIGHT_NEIGHBORS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        color2 = pattern.get_cell(nr, nc)
                        if color2 > 0 and color1 != color2:
                            edge_count += 1
        return edge_count // 2

    def _shuffle(self, values: List[int]) -> None:
        for i in range(len(values) - 1, 0, -1):
            j = self._random_int(i + 1)
            values[i], values[j] = values[j], values[i]

    def _assign_colors_to_pattern(
        self,
        spatial_pattern: PatternGrid,
        block_count: int,
        color_count: int,
        strategy: str = "random",
    ) -> PatternGrid:
        positions = spatial_pattern.get_occupied_positions()
        active_count = len(positions)
        pattern = spatial_pattern.copy()
        if active_count == 0:
            return pattern

        base_count = active_count // color_count
        remainder = active_count % color_count
        counts = [base_count + (1 if color <= remainder else 0) for color in range(1, color_count + 1)]

        if strategy == "chunked":
            remaining: OrderedDict[str, None] = OrderedDict((f"{r},{c}", None) for r, c in positions)
            for color_idx, count in enumerate(counts):
                if count == 0 or not remaining:
                    continue
                start_key = next(iter(remaining), None)
                if start_key is None:
                    return pattern
                queue: deque[str] = deque([start_key])
                visited_local: set[str] = set()
                color = color_idx + 1
                assigned = 0
                while queue and assigned < count:
                    key = queue.popleft()
                    if key not in remaining or key in visited_local:
                        continue
                    visited_local.add(key)
                    r_str, c_str = key.split(",")
                    r, c = int(r_str, 10), int(c_str, 10)
                    pattern.set_cell(r, c, color)
                    remaining.pop(key, None)
                    assigned += 1
                    for dr, dc in EIGHT_NEIGHBORS:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                            neighbor_key = f"{nr},{nc}"
                            if neighbor_key in remaining and neighbor_key not in visited_local:
                                queue.append(neighbor_key)
            if remaining:
                for key in list(remaining.keys()):
                    r_str, c_str = key.split(",")
                    r, c = int(r_str, 10), int(c_str, 10)
                    pattern.set_cell(r, c, color_count)
            return pattern

        colors: List[int] = []
        for color_idx, count in enumerate(counts):
            color = color_idx + 1
            colors.extend([color] * count)
        self._shuffle(colors)

        for idx, (r, c) in enumerate(positions):
            if idx >= len(colors):
                break
            pattern.set_cell(r, c, colors[idx])

        return pattern

    def _build_color_assignments(
        self,
        spatial_pattern: PatternGrid,
        block_count: int,
        color_count: int,
        sample_size: int = COLOR_SAMPLES,
    ) -> List[Dict[str, object]]:
        assignments: List[Dict[str, object]] = []
        capacity = self._count_occupied_adjacencies(spatial_pattern)
        capacity = max(capacity, 1)
        for _ in range(sample_size):
            colored_pattern = self._assign_colors_to_pattern(spatial_pattern, block_count, color_count)
            edges = self._count_different_color_edges(colored_pattern)
            ratio = edges / capacity
            assignments.append({"pattern": colored_pattern, "edges": edges, "ratio": ratio, "capacity": capacity, "priority": 0})

        chunked_pattern = self._assign_colors_to_pattern(spatial_pattern, block_count, color_count, "chunked")
        chunked_edges = self._count_different_color_edges(chunked_pattern)
        assignments.append(
            {
                "pattern": chunked_pattern,
                "edges": chunked_edges,
                "ratio": chunked_edges / capacity,
                "capacity": capacity,
                "priority": -1,
            }
        )

        assignments.sort(key=lambda item: (item["edges"], item.get("priority", 0)))
        return assignments

    @staticmethod
    def _annotate_percentiles(entries: List[Dict[str, object]], metric_key: str, percentile_key: str) -> None:
        if not entries:
            return
        sorted_entries = sorted(entries, key=lambda entry: float(entry[metric_key]))
        if len(sorted_entries) == 1:
            sorted_entries[0][percentile_key] = 0.0
            return
        denom = len(sorted_entries) - 1
        for index, entry in enumerate(sorted_entries):
            entry[percentile_key] = (index / denom) * 100.0

    def _select_color_assignment(
        self,
        assignments: List[Dict[str, object]],
        percentile: float,
    ) -> Dict[str, object]:
        if not assignments:
            return {"pattern": None, "edges": 0, "achievedPercentile": 0.0}
        if len(assignments) == 1:
            entry = assignments[0]
            return {"pattern": entry["pattern"], "edges": entry["edges"], "ratio": entry.get("ratio", 0.0), "capacity": entry.get("capacity", 1), "achievedPercentile": 0.0}

        min_edges = assignments[0]["edges"]
        max_edges = assignments[-1]["edges"]
        min_ratio = assignments[0].get("ratio", 0.0)
        max_ratio = assignments[-1].get("ratio", 0.0)
        if max_edges == min_edges or math.isclose(max_ratio, min_ratio):
            entry = assignments[0]
            return {
                "pattern": entry["pattern"],
                "edges": entry["edges"],
                "ratio": entry.get("ratio", 0.0),
                "capacity": entry.get("capacity", 1),
                "achievedPercentile": 0.0,
            }

        normalized_target = max(0.0, min(1.0, float(percentile) / 100.0))
        desired_ratio = min_ratio + normalized_target * (max_ratio - min_ratio)
        best = assignments[0]
        best_diff = abs(best.get("ratio", 0.0) - desired_ratio)
        for candidate in assignments[1:]:
            diff = abs(candidate.get("ratio", 0.0) - desired_ratio)
            if diff < best_diff:
                best = candidate
                best_diff = diff
        achieved_normalized = (best.get("ratio", 0.0) - min_ratio) / (max_ratio - min_ratio)
        achieved_percentile = max(0.0, min(100.0, achieved_normalized * 100.0))
        return {
            "pattern": best["pattern"],
            "edges": best["edges"],
            "ratio": best.get("ratio", 0.0),
            "capacity": best.get("capacity", 1),
            "achievedPercentile": achieved_percentile,
        }

    def _analyze_color_components(self, pattern: PatternGrid, color_count: int) -> Dict[str, object]:
        visited = [[False for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        counts = [0 for _ in range(color_count + 1)]

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                color = pattern.get_cell(r, c)
                if color > 0 and not visited[r][c]:
                    counts[color] += 1
                    queue: deque[Tuple[int, int]] = deque([(r, c)])
                    visited[r][c] = True
                    while queue:
                        cr, cc = queue.popleft()
                        for dr, dc in EIGHT_NEIGHBORS:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                                if not visited[nr][nc] and pattern.get_cell(nr, nc) == color:
                                    visited[nr][nc] = True
                                    queue.append((nr, nc))

        max_components = 0
        for color in range(1, color_count + 1):
            if counts[color] > max_components:
                max_components = counts[color]

        return {"counts": counts, "maxComponents": max_components}

    # --- Sampling logic -------------------------------------------------

    def _sample_joint_dataset(
        self,
        block_count: int,
        color_count: int,
        seed_offset: int = 3000,
    ) -> List[Dict[str, object]]:
        cache_key = f"{self.grid_size}-{block_count}-{color_count}-{'seed' if self.use_fixed_seed else 'random'}-{self.seed_value if self.use_fixed_seed else 0}"
        cached = self.joint_dataset_cache.get(cache_key)
        if cached is not None:
            return cached

        self._prepare_seed(seed_offset)
        patterns: List[Dict[str, object]] = []
        attempts = 0
        target_patterns = min(JOINT_SPATIAL_SAMPLES, SPATIAL_SAMPLES)

        while len(patterns) < target_patterns and attempts < SPATIAL_ATTEMPTS_MAX:
            attempts += 1
            spatial_pattern = self._generate_connected_spatial_pattern(block_count)
            if self._has_symmetry(spatial_pattern) or not self._is_connected(spatial_pattern):
                continue
            entropy = self._calculate_spatial_entropy(spatial_pattern)
            adjacency = self._count_occupied_adjacencies(spatial_pattern)
            patterns.append({"pattern": spatial_pattern, "entropy": entropy, "adjacency": adjacency})

        dataset: List[Dict[str, object]] = []
        if not patterns:
            fallback_pattern = self._generate_connected_spatial_pattern(block_count)
            fallback_entropy = self._calculate_spatial_entropy(fallback_pattern)
            assignments = self._build_color_assignments(
                fallback_pattern,
                block_count,
                color_count,
                JOINT_COLOR_ASSIGNMENTS,
            )
            for assignment in assignments:
                dataset.append(
                    {
                        "pattern": assignment["pattern"],
                        "entropy": fallback_entropy,
                        "edges": assignment["edges"],
                        "capacity": assignment.get("capacity", 1),
                        "ratio": assignment.get("ratio", 0.0),
                    }
                )
        else:
            assignments_per_pattern = JOINT_COLOR_ASSIGNMENTS
            for index, entry in enumerate(patterns):
                self._prepare_seed(seed_offset + 100 + index)
                assignments = self._build_color_assignments(
                    entry["pattern"],
                    block_count,
                    color_count,
                    assignments_per_pattern,
                )
                step = max(1, len(assignments) // max(1, assignments_per_pattern))
                for i in range(0, len(assignments), step):
                    assignment = assignments[i]
                    dataset.append(
                        {
                            "pattern": assignment["pattern"],
                            "entropy": entry["entropy"],
                            "edges": assignment["edges"],
                            "capacity": assignment.get("capacity", 1),
                            "ratio": assignment.get("ratio", 0.0),
                        }
                    )
                if assignments:
                    first_assignment = assignments[0]
                    last_assignment = assignments[-1]
                    dataset.append(
                        {
                            "pattern": first_assignment["pattern"],
                            "entropy": entry["entropy"],
                            "edges": first_assignment["edges"],
                            "capacity": first_assignment.get("capacity", 1),
                            "ratio": first_assignment.get("ratio", 0.0),
                        }
                    )
                    if last_assignment is not first_assignment:
                        dataset.append(
                            {
                                "pattern": last_assignment["pattern"],
                                "entropy": entry["entropy"],
                                "edges": last_assignment["edges"],
                                "capacity": last_assignment.get("capacity", 1),
                                "ratio": last_assignment.get("ratio", 0.0),
                            }
                        )

        if len(dataset) > 15000:
            dataset = dataset[:15000]

        self._annotate_percentiles(dataset, "entropy", "spatialPercentile")
        self._annotate_percentiles(dataset, "ratio", "colorPercentile")
        self.joint_dataset_cache[cache_key] = dataset
        return dataset

    def _sample_joint_points(
        self,
        block_count: int,
        color_count: int,
        sample_count: int,
    ) -> List[Dict[str, object]]:
        dataset = self._sample_joint_dataset(block_count, color_count)
        if not dataset:
            return []
        target = max(1, min(sample_count, len(dataset)))
        indices = list(range(len(dataset)))
        self._prepare_seed(4200)
        self._shuffle(indices)
        samples: List[Dict[str, object]] = []
        for idx in indices[:target]:
            entry = dataset[idx]
            samples.append(
                {
                    "entropy": entry["entropy"],
                    "edges": entry["edges"],
                    "percentile": entry.get("colorPercentile"),
                    "spatialPercentile": entry.get("spatialPercentile"),
                    "maxEdges": entry.get("capacity") or self._count_occupied_adjacencies(entry["pattern"]),
                    "isRandom": True,
                }
            )
        return samples

    def _generate_joint_selection(
        self,
        block_count: int,
        color_count: int,
        spatial_percentile: float,
        color_percentile: float,
        seed_offset: int = 0,
    ) -> PatternMetrics:
        start_time = self._now()
        dataset = self._sample_joint_dataset(block_count, color_count, 4000)
        if not dataset:
            return self._generate_at_percentiles(
                block_count,
                color_count,
                spatial_percentile,
                color_percentile,
                seed_offset,
                mode="conditional",
            )

        target_spatial = max(0.0, min(100.0, spatial_percentile))
        target_color = max(0.0, min(100.0, color_percentile))

        best = dataset[0]
        best_score = float("inf")
        for entry in dataset:
            ds = abs(float(entry.get("spatialPercentile", 0.0)) - target_spatial)
            dc = abs(float(entry.get("colorPercentile", 0.0)) - target_color)
            score = ds + dc
            if score < best_score:
                best = entry
                best_score = score

        max_possible_edges = self._count_occupied_adjacencies(best["pattern"])
        color_components = self._analyze_color_components(best["pattern"], color_count)
        elapsed = self._now() - start_time
        return PatternMetrics(
            pattern=best["pattern"],
            spatial_entropy=float(best["entropy"]),
            edges=int(best["edges"]),
            max_possible_edges=int(max_possible_edges),
            is_connected=self._is_connected(best["pattern"]),
            generation_time=elapsed,
            achieved_spatial_percentile=float(best.get("spatialPercentile", 0.0)),
            achieved_color_percentile=float(best.get("colorPercentile", 0.0)),
            max_color_components=int(color_components["maxComponents"]),
            block_count=block_count,
            color_count=color_count,
            spatial_target=spatial_percentile,
            color_target=color_percentile,
        )

    def _sample_conditional_percentiles(
        self,
        block_count: int,
        color_count: int,
        sample_count: int,
        seed_offset: int = 1000,
    ) -> List[Dict[str, object]]:
        effective_count = max(sample_count, 0)
        samples: List[Dict[str, object]] = []
        patterns: List[Dict[str, object]] = []
        max_attempts = max(effective_count, 1) * 10
        attempts = 0
        self._prepare_seed(seed_offset)

        while len(patterns) < effective_count and attempts < max_attempts:
            attempts += 1
            spatial_pattern = self._generate_connected_spatial_pattern(block_count)
            if self._has_symmetry(spatial_pattern) or not self._is_connected(spatial_pattern):
                continue
            entropy = self._calculate_spatial_entropy(spatial_pattern)
            adjacency = self._count_occupied_adjacencies(spatial_pattern)
            patterns.append({"pattern": spatial_pattern, "entropy": entropy, "adjacency": adjacency})

        patterns.sort(key=lambda entry: (entry["entropy"], entry["adjacency"]))

        for index, entry in enumerate(patterns):
            self._prepare_seed(seed_offset + index + 1)
            assignments = self._build_color_assignments(
                entry["pattern"],
                block_count,
                color_count,
                min(SCATTER_COLOR_SAMPLES, COLOR_SAMPLES),
            )
            for percentile in (1, 50, 99):
                selection = self._select_color_assignment(assignments, percentile)
                samples.append(
                    {
                        "entropy": entry["entropy"],
                        "edges": selection["edges"],
                        "percentile": percentile,
                        "maxEdges": selection.get("capacity"),
                        "isRandom": True,
                    }
                )

        return samples

    def _generate_at_percentiles(
        self,
        block_count: int,
        color_count: int,
        spatial_percentile: float,
        color_percentile: float,
        seed_offset: int = 0,
        mode: Optional[str] = None,
    ) -> PatternMetrics:
        target_mode = mode or self.percentile_mode
        if target_mode == "joint":
            return self._generate_joint_selection(
                block_count,
                color_count,
                spatial_percentile,
                color_percentile,
                seed_offset,
            )

        start_time = self._now()
        self._prepare_seed(seed_offset)
        spatial_patterns: List[Dict[str, object]] = []
        attempts = 0

        while len(spatial_patterns) < SPATIAL_SAMPLES and attempts < SPATIAL_ATTEMPTS_MAX:
            attempts += 1
            pattern = self._generate_connected_spatial_pattern(block_count)
            if not self._has_symmetry(pattern) and self._is_connected(pattern):
                entropy = self._calculate_spatial_entropy(pattern)
                adjacency = self._count_occupied_adjacencies(pattern)
                spatial_patterns.append({"pattern": pattern, "entropy": entropy, "adjacency": adjacency})

        if not spatial_patterns:
            fallback_pattern = self._generate_connected_spatial_pattern(block_count)
            spatial_patterns.append(
                {
                    "pattern": fallback_pattern,
                    "entropy": self._calculate_spatial_entropy(fallback_pattern),
                    "adjacency": self._count_occupied_adjacencies(fallback_pattern),
                }
            )
            print("No asymmetric connected patterns generated; using fallback.", file=sys.stderr)
        elif len(spatial_patterns) < min(100, SPATIAL_SAMPLES):
            print("Could not generate enough valid patterns, using what we have", file=sys.stderr)

        spatial_patterns.sort(key=lambda entry: (entry["entropy"], entry["adjacency"]))

        index, achieved_spatial = self._get_percentile_selection(len(spatial_patterns), spatial_percentile)
        selected_spatial_pattern = spatial_patterns[index]["pattern"]

        color_assignments = self._build_color_assignments(selected_spatial_pattern, block_count, color_count, COLOR_SAMPLES)
        color_selection = self._select_color_assignment(color_assignments, color_percentile)
        final_pattern = color_selection.get("pattern") or color_assignments[0]["pattern"]

        elapsed = self._now() - start_time
        max_possible_edges = self._count_occupied_adjacencies(final_pattern)
        color_components = self._analyze_color_components(final_pattern, color_count)

        return PatternMetrics(
            pattern=final_pattern,
            spatial_entropy=float(spatial_patterns[index]["entropy"]),
            edges=int(color_selection["edges"]),
            max_possible_edges=int(max_possible_edges),
            is_connected=self._is_connected(final_pattern),
            generation_time=elapsed,
            achieved_spatial_percentile=float(achieved_spatial),
            achieved_color_percentile=float(color_selection["achievedPercentile"]),
            max_color_components=int(color_components["maxComponents"]),
            block_count=block_count,
            color_count=color_count,
            spatial_target=spatial_percentile,
            color_target=color_percentile,
        )


def generate_rows_and_scatter(
    generator: PatternGenerator,
    pairs: Sequence[Tuple[int, int]],
    percentiles: Sequence[float],
) -> Tuple[List[List[PatternMetrics]], List[ScatterPoint]]:
    rows: List[List[PatternMetrics]] = []
    scatter_points: List[ScatterPoint] = []
    patterns_per_row = max(1, len(percentiles))

    for pair_index, (block_count, color_count) in enumerate(pairs):
        row_results: List[PatternMetrics] = []
        for col_index, percentile in enumerate(percentiles):
            result = generator._generate_at_percentiles(
                block_count,
                color_count,
                spatial_percentile=percentile,
                color_percentile=percentile,
                seed_offset=pair_index * patterns_per_row + col_index + 1,
            )
            row_results.append(result)
            label = f"L{block_count}/K{color_count} • S{percentile}/C{percentile}"
            scatter_points.append(
                ScatterPoint(
                    entropy=result.spatial_entropy,
                    edges=result.edges,
                    max_edges=float(result.max_possible_edges),
                    label=label,
                    is_random=False,
                    percentile=float(percentile),
                    spatial_percentile=result.achieved_spatial_percentile,
                    color_percentile=result.achieved_color_percentile,
                    color_index=col_index,
                    pair_index=pair_index,
                )
            )

        if generator.percentile_mode == "joint":
            random_samples = generator._sample_joint_points(
                block_count,
                color_count,
                generator.random_sample_count,
            )
        else:
            random_samples = generator._sample_conditional_percentiles(
                block_count,
                color_count,
                generator.random_sample_count,
            )
        for sample in random_samples:
            percentile = sample.get("percentile")
            spatial_pct = sample.get("spatialPercentile")
            color_pct = sample.get("colorPercentile")
            pattern_capacity = sample.get("maxEdges")
            scatter_points.append(
                ScatterPoint(
                    entropy=float(sample["entropy"]),
                    edges=float(sample["edges"]),
                    max_edges=float(pattern_capacity) if pattern_capacity is not None else None,
                    label="",
                    is_random=True,
                    percentile=float(percentile) if percentile is not None else None,
                    spatial_percentile=float(spatial_pct) if spatial_pct is not None else None,
                    color_percentile=float(color_pct) if color_pct is not None else None,
                    pair_index=pair_index,
                )
            )
        rows.append(row_results)

    return rows, scatter_points


def _ordinal(value: float) -> str:
    rounded = int(round(value))
    suffix = "th"
    if 10 <= rounded % 100 <= 20:
        suffix = "th"
    else:
        last_digit = rounded % 10
        if last_digit == 1:
            suffix = "st"
        elif last_digit == 2:
            suffix = "nd"
        elif last_digit == 3:
            suffix = "rd"
    return f"{rounded}{suffix}"


def plot_scatter(
    ax: plt.Axes,
    points: Sequence[ScatterPoint],
    mode: str,
    title: str,
    *,
    show_xlabel: bool,
) -> None:
    if not points:
        ax.set_visible(False)
        return

    def _ratio(point: ScatterPoint) -> float:
        capacity = point.max_edges
        if capacity is None or capacity <= 0:
            return 0.0
        value = float(point.edges) / float(capacity)
        return max(0.0, min(1.0, value))

    raw_entropies = np.array([float(p.entropy) for p in points], dtype=float)
    raw_ratios = np.array([_ratio(p) for p in points], dtype=float)
    min_entropy = float(raw_entropies.min())
    max_entropy = float(raw_entropies.max())
    min_ratio = float(raw_ratios.min())
    max_ratio = float(raw_ratios.max())

    entropy_range = max(max_entropy - min_entropy, 1e-6)
    ratio_range = max(max_ratio - min_ratio, 1e-6)

    pad_x = 0.06
    pad_y = 0.06

    def normalize(values: Sequence[float], lo: float, rng: float, pad: float) -> np.ndarray:
        return np.clip((np.asarray(values, dtype=float) - lo) / rng, -pad, 1.0 + pad)

    norm_entropies = normalize(raw_entropies, min_entropy, entropy_range, pad_x)
    norm_ratios = normalize(raw_ratios, min_ratio, ratio_range, pad_y)

    ax.set_xlim(-pad_x, 1.0 + pad_x)
    ax.set_ylim(-pad_y, 1.0 + pad_y)

    ax.set_xlabel("Spatial Entropy" if show_xlabel else "", fontsize=10 if show_xlabel else 0)
    ax.set_ylabel("Color Mix Ratio", fontsize=10)
    ax.tick_params(
        axis="both",
        which="both",
        bottom=True,
        top=False,
        left=True,
        right=False,
        labelsize=9,
        colors="#333333",
        length=3.5,
        width=0.8,
    )
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#fafafa")
    ax.grid(False)
    ax.set_title(title, fontsize=10, pad=6, color="#333333")

    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
    ]

    random_points = [p for p in points if p.is_random]
    main_points = [p for p in points if not p.is_random]

    density_plotted = False
    density_candidates: List[ScatterPoint] = []
    if random_points:
        density_candidates.extend(random_points)
    if main_points:
        # Replicate percentile picks heavily so their regions register in the density map.
        density_candidates.extend(main_points * 25)
    if len(density_candidates) >= 10:
        xs = normalize([p.entropy for p in density_candidates], min_entropy, entropy_range, pad_x)
        ys = normalize([_ratio(p) for p in density_candidates], min_ratio, ratio_range, pad_y)
        unique = len({(round(x, 4), round(y, 4)) for x, y in zip(xs, ys)})
        if unique > 5:
            bins = 140
            hist, x_edges, y_edges = np.histogram2d(
                xs,
                ys,
                bins=bins,
                range=[[-pad_x, 1.0 + pad_x], [-pad_y, 1.0 + pad_y]],
            )
            if hist.max() > 0:
                sigma = max(2.5, bins / 70)
                radius = max(1, int(3 * sigma))
                kernel_range = np.arange(-radius, radius + 1, dtype=float)
                kernel = np.exp(-0.5 * (kernel_range / sigma) ** 2)
                kernel /= kernel.sum()
                smoothed = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=hist)
                smoothed = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=smoothed)
                smoothed /= smoothed.max()

                x_centers = (x_edges[:-1] + x_edges[1:]) / 2
                y_centers = (y_edges[:-1] + y_edges[1:]) / 2
                X, Y = np.meshgrid(x_centers, y_centers, indexing="xy")
                Z = smoothed.T

                base_levels = np.concatenate(
                    (
                        np.array([1e-5, 5e-5, 1e-4, 5e-4, 1e-3]),
                        np.linspace(0.005, 0.9, 16),
                        np.array([0.93, 0.96, 0.98, 0.995]),
                    )
                )
                ax.contourf(
                    X,
                    Y,
                    Z,
                    levels=base_levels,
                    cmap="Greys",
                    alpha=0.65,
                    antialiased=True,
                    extend="both",
                )
                ax.contour(
                    X,
                    Y,
                    Z,
                    levels=base_levels[8:],
                    colors="#555555",
                    linewidths=0.45,
                    alpha=0.45,
                )
                density_plotted = True

    if not density_plotted and random_points:
        xs = normalize([p.entropy for p in random_points], min_entropy, entropy_range, pad_x)
        ys = normalize([_ratio(p) for p in random_points], min_ratio, ratio_range, pad_y)
        ax.scatter(xs, ys, s=12, c="#bbbbbb", alpha=0.35, linewidths=0, zorder=1)

    if main_points:
        legend_entries: List[str] = []
        for point in main_points:
            color = "#4c6ef5"
            x = float(normalize([point.entropy], min_entropy, entropy_range, pad_x)[0])
            y = float(normalize([_ratio(point)], min_ratio, ratio_range, pad_y)[0])
            ax.scatter(
                [x],
                [y],
                s=110,
                c=color,
                edgecolors="black",
                linewidths=1.2,
                zorder=3,
            )
            marker_label = point.label.split(" • ")[-1].split("/")[-1]
            marker_label = marker_label.replace("C", "").rstrip("%") + "%"
            ax.text(
                x,
                y + 0.02,
                marker_label,
                fontsize=8,
                color="#333333",
                ha="center",
                va="bottom",
                zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
            )
            legend_entries.append(point.label)
        # Legend callout intentionally omitted; density shade + markers are sufficient.


def build_representative_figure(
    rows: Sequence[Sequence[PatternMetrics]],
    scatter_points: Sequence[ScatterPoint],
    percentiles: Sequence[float],
    pairs: Sequence[Tuple[int, int]],
    *,
    mode: str,
    title: str,
    output_path: Path,
) -> Path:
    n_rows = len(rows)
    n_cols = (len(percentiles) or 1) + 1
    width = 2.8 + max(1, len(percentiles)) * 2.1
    height = max(3.6, n_rows * 2.8)

    fig_width = max(3.0, n_cols * 2.6)
    fig_height = max(3.6, n_rows * 2.6)
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(
        n_rows,
        n_cols,
        figure=fig,
        width_ratios=[1.0] * n_cols,
        hspace=0.35,
        wspace=0.12,
    )

    available_colors: deque[str] = deque(CARPET_COLORS)
    total_unique_colors = len(CARPET_COLORS)

    for row_idx, (row, pair) in enumerate(zip(rows, pairs)):
        block_count, color_count = pair
        pair_points = [point for point in scatter_points if point.pair_index == row_idx]
        ax_pair_scatter = fig.add_subplot(gs[row_idx, 0])
        scatter_title = fr"$L={block_count}$, $K={color_count}$"
        plot_scatter(
            ax_pair_scatter,
            pair_points,
            mode,
            scatter_title,
            show_xlabel=(row_idx == n_rows - 1),
        )

        palette: List[str] = []
        allow_duplicates = color_count > total_unique_colors

        while len(palette) < color_count:
            if not available_colors:
                if allow_duplicates:
                    available_colors = deque(CARPET_COLORS)
                else:
                    refreshed = [color for color in CARPET_COLORS if color not in palette]
                    available_colors = deque(refreshed)
                    if not available_colors:
                        # All colors already used in this row; duplicates are unavoidable now.
                        allow_duplicates = True
                        available_colors = deque(CARPET_COLORS)
            if not available_colors:
                break

            candidate = available_colors.popleft()
            if not allow_duplicates and candidate in palette:
                continue
            palette.append(candidate)

        color_list = [EMPTY_COLOR, *palette]
        cmap = ListedColormap(color_list)
        bounds = np.arange(len(color_list) + 1) - 0.5
        norm = BoundaryNorm(bounds, cmap.N)

        for col_idx, metrics in enumerate(row):
            ax = fig.add_subplot(gs[row_idx, col_idx + 1])
            data = np.array(metrics.pattern.grid)
            ax.imshow(data, cmap=cmap, norm=norm, interpolation="none", origin="upper")
            grid_size = metrics.pattern.grid_size
            ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
            ax.grid(which="minor", color="#ffffff", linewidth=0.8)
            ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
            ax.set_facecolor("#fdfdfd")
            ax.set_aspect("equal")

            label = _ordinal(percentiles[col_idx]) if col_idx < len(percentiles) else _ordinal(metrics.spatial_target)
            if mode == "joint":
                label = f"{label} Joint Percentile"
            ax.set_title(label, fontsize=10, pad=6, color="#333333")

    fig.suptitle(title, fontsize=14, y=0.99, color="#222222")
    fig.savefig(output_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=DEFAULT_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _parse_pair(text: str) -> Tuple[int, int]:
    for sep in (",", ":", "x", "X", "-"):
        if sep in text:
            left, right = text.split(sep, 1)
            break
    else:
        parts = text.split()
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"Expected pair formatted as L,K but got '{text}'")
        left, right = parts
    try:
        l_val = int(left)
        k_val = int(right)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer pair '{text}'") from exc
    if l_val <= 0 or k_val <= 0:
        raise argparse.ArgumentTypeError("L and K must be positive integers")
    return l_val, k_val


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate representative carpet pattern figure")
    parser.add_argument(
        "--pairs",
        nargs="+",
        type=_parse_pair,
        default=list(DEFAULT_PAIRS),
        help="List of L,K pairs for the 5x5 grid (e.g. 7,2 10,3).",
    )
    parser.add_argument(
        "--percentiles",
        nargs="+",
        type=float,
        default=list(DEFAULT_PERCENTILES),
        help="Representative percentiles (applied to both spatial and color).",
    )
    parser.add_argument(
        "--mode",
        choices=("conditional", "joint"),
        default="joint",
        help="Percentile selection mode (default joint).",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=DEFAULT_GRID_SIZE,
        help="Grid size N for the NxN pattern (default 5).",
    )
    parser.add_argument(
        "--random-samples",
        type=int,
        default=10_000,
        help="Random sample count for scatter plot background (default 10000).",
    )
    parser.add_argument(
        "--use-fixed-seed",
        dest="use_fixed_seed",
        action="store_true",
        help="Retained for backward compatibility; the script always uses a fixed seed.",
    )
    parser.add_argument(
        "--no-fixed-seed",
        dest="use_fixed_seed",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed value to use; defaults to 1695856744 when omitted.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Representative Pattern Sets",
        help="Figure title (default 'Representative Pattern Sets').",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path; defaults to manuscript_figures/Figure02_Patterns/Figure02_Patterns.png.",
    )
    parser.add_argument(
        "--no-show",  # placeholder for parity with other scripts if user wants to suppress printing
        dest="show_summary",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(show_summary=True, use_fixed_seed=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.grid_size <= 0:
        raise ValueError("Grid size must be positive")

    user_provided_seed = args.seed is not None
    if user_provided_seed:
        seed_value = args.seed & 0xFFFFFFFF
    else:
        seed_value = 1695856744 & 0xFFFFFFFF

    # Always operate with a fixed seed so results are reproducible given the number.
    args.use_fixed_seed = True

    # Synchronize Python and NumPy RNGs in case any code path uses them directly.
    random.seed(seed_value)
    np.random.seed(seed_value)


    percentiles = list(args.percentiles)
    if not percentiles:
        raise ValueError("At least one percentile must be provided")

    for value in percentiles:
        if value < 0 or value > 100:
            raise ValueError("Percentiles must be between 0 and 100")

    pairs = list(args.pairs)
    if not pairs:
        raise ValueError("At least one L,K pair must be specified")

    for l_val, _k_val in pairs:
        if l_val > args.grid_size * args.grid_size:
            raise ValueError(f"Block count L={l_val} exceeds grid capacity {args.grid_size ** 2}")

    generator = PatternGenerator(
        grid_size=args.grid_size,
        use_fixed_seed=args.use_fixed_seed,
        seed_value=seed_value,
        random_sample_count=args.random_samples,
        percentile_mode=args.mode,
    )

    rows, scatter_points = generate_rows_and_scatter(generator, pairs, percentiles)

    default_dir = figure_output_dir("Fig02")
    default_stem = figure_output_stem("Fig02")
    if args.output is None:
        output_dir = default_dir
        output_path = output_dir / f"{default_stem}.png"
    else:
        provided = args.output
        if provided.suffix:
            output_path = provided
            output_dir = provided.parent
        else:
            output_dir = provided
            output_path = output_dir / f"{default_stem}.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    title = args.title
    build_representative_figure(
        rows,
        scatter_points,
        percentiles,
        pairs,
        mode=args.mode,
        title=title,
        output_path=output_path,
    )

    if args.show_summary:
        print(f"Wrote representative pattern figure to {output_path} (seed={seed_value})")
        if not user_provided_seed:
            cmd = f"python src/figures/MakeFigure02.py --seed {seed_value}"
            print(f"Reproduce with: {cmd}")


if __name__ == "__main__":  # pragma: no cover
    main()
