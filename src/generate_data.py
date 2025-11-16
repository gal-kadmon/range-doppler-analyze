# generate_data.py
# -----------------------------------------------------------------------------
# Synthetic Range–Doppler data generator.
#
# This module creates two binary Range–Doppler matrices (matrix1, matrix2) with:
#   • Unique targets in matrix1
#   • Unique targets in matrix2
#   • Shared targets appearing in both matrices
#
# Each target is a 4-connected shape (no diagonal-only links) constructed over
# a binary grid. Targets are written as 1s in otherwise-zero matrices.
#
# The function returns a Process_Data_Unit (PDU) containing:
#   • Two tensors: "matrix1", "matrix2"
#   • Base attributes including a UUID and metadata describing the generation
#   • A text log file recording all generated target coordinates
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import uuid
import random
from typing import List, Tuple, Optional, Dict

import numpy as np

from process_data_unit import (
    Process_Data_Unit,
    ndarray_data,
    base_data_attr,
)


Coord = Tuple[int, int]
CoordList = List[Coord]


def _neighbors4(r: int, c: int) -> List[Coord]:
    """Return the 4-connected neighbor coordinates around (r, c)."""
    return [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]


def _in_bounds(r: int, c: int, shape: Tuple[int, int]) -> bool:
    """Check if (r, c) lies within the given matrix shape."""
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _try_add_pixel(
    coords: set[Coord],
    frontier: List[Coord],
    r: int,
    c: int,
    shape: Tuple[int, int],
    occupied: Optional[np.ndarray] = None,
) -> bool:
    """
    Attempt to add pixel (r, c) to the current connected shape.

    A pixel is added only if:
      • It is inside the matrix bounds
      • It is not already in the shape
      • It is not already occupied (if an occupancy mask is provided)
    """
    if not _in_bounds(r, c, shape):
        return False
    if (r, c) in coords:
        return False
    if occupied is not None and occupied[r, c]:
        return False
    coords.add((r, c))
    frontier.append((r, c))
    return True


def generate_connected_shape(
    shape: Tuple[int, int],
    size: int,
    rng: random.Random,
    occupied: Optional[np.ndarray] = None,
    max_tries: int = 10_000,
) -> CoordList:
    """
    Generate a single 4-connected shape of exactly `size` pixels.

    The shape is grown by a randomized region-growing process using only
    4-connected neighbors. If an occupancy mask is provided, the shape is
    constrained to non-occupied cells.

    Raises
    ------
    ValueError:
        If a valid shape cannot be constructed after `max_tries` attempts.
    """
    rows, cols = shape
    tries = 0

    while tries < max_tries:
        tries += 1

        # Choose a random starting pixel
        sr = rng.randrange(rows)
        sc = rng.randrange(cols)
        if occupied is not None and occupied[sr, sc]:
            continue

        coords: set[Coord] = {(sr, sc)}
        frontier: List[Coord] = [(sr, sc)]

        # Grow until requested size is reached or no valid expansion remains
        while len(coords) < size and frontier:
            r, c = rng.choice(frontier)
            neighbors = _neighbors4(r, c)
            rng.shuffle(neighbors)

            progressed = False
            for nr, nc in neighbors:
                if _try_add_pixel(coords, frontier, nr, nc, shape, occupied):
                    progressed = True
                    if len(coords) >= size:
                        break

            # Remove frontier pixels that cannot be expanded further
            new_frontier = []
            for fr, fc in frontier:
                has_free_neighbor = False
                for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    rr = fr + dr
                    cc = fc + dc
                    if (
                        _in_bounds(rr, cc, shape)
                        and (rr, cc) not in coords
                        and (occupied is None or not occupied[rr, cc])
                    ):
                        has_free_neighbor = True
                        break
                if has_free_neighbor:
                    new_frontier.append((fr, fc))
            frontier = new_frontier

            if not progressed and not frontier:
                break

        if len(coords) == size:
            return list(coords)

    raise ValueError("Unable to generate connected shape with requested size.")


def _build_partial_overlap(
    base_coords: CoordList,
    shape: Tuple[int, int],
    rng: random.Random,
    occ_m2: np.ndarray,
    keep_ratio_range: Tuple[float, float] = (0.4, 0.8),
    extra_cells_range: Tuple[int, int] = (0, 5),
) -> CoordList:
    """
    Build a second shape for matrix2 that partially overlaps the base shape
    for matrix1, while preserving 4-connectivity.

    The procedure:
      • Keep a random subset of base_coords as the intersection region.
      • Grow additional pixels (4-connected) that are outside the base shape
        and not occupied yet in matrix2.
    """
    rows, cols = shape
    base_set = set(base_coords)

    keep_ratio = rng.uniform(*keep_ratio_range)
    keep_count = max(1, int(round(keep_ratio * len(base_coords))))
    overlap_subset = set(rng.sample(base_coords, keep_count))

    coords_m2: set[Coord] = set(overlap_subset)
    frontier = list(overlap_subset) if overlap_subset else [rng.choice(base_coords)]

    extra_cells = rng.randint(*extra_cells_range)

    while len(coords_m2) < keep_count + extra_cells and frontier:
        r, c = rng.choice(frontier)
        neighbors = _neighbors4(r, c)
        rng.shuffle(neighbors)

        progressed = False
        for nr, nc in neighbors:
            if not _in_bounds(nr, nc, (rows, cols)):
                continue
            if (nr, nc) in coords_m2:
                continue
            if occ_m2[nr, nc]:
                continue
            if (nr, nc) in base_set:
                continue

            coords_m2.add((nr, nc))
            frontier.append((nr, nc))
            progressed = True
            if len(coords_m2) >= keep_count + extra_cells:
                break

        # Remove dead-end frontier pixels
        new_frontier = []
        for fr, fc in frontier:
            has_free_neighbor = False
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                rr = fr + dr
                cc = fc + dc
                if (
                    _in_bounds(rr, cc, (rows, cols))
                    and (rr, cc) not in coords_m2
                    and not occ_m2[rr, cc]
                    and (rr, cc) not in base_set
                ):
                    has_free_neighbor = True
                    break
            if has_free_neighbor:
                new_frontier.append((fr, fc))
        frontier = new_frontier

        if not progressed and not frontier:
            break

    return list(coords_m2)


def generate_synthetic_rd_data(
    matrix_shape: Tuple[int, int] = (32, 1024),
    num_unique_m1: int = 6,
    num_unique_m2: int = 6,
    num_shared: int = 4,
    target_size_range: Tuple[int, int] = (3, 12),
    partial_keep_ratio_range: Tuple[float, float] = (0.4, 0.8),
    partial_extra_cells_range: Tuple[int, int] = (0, 5),
    seed: Optional[int] = None,
    log_filename: str = "synthetic_generation_log.txt",
) -> Process_Data_Unit:
    """
    Generate two synthetic Range–Doppler binary matrices and wrap them in a PDU.

    Parameters
    ----------
    matrix_shape : (int, int)
        Shape of both matrices (DP, RG).
    num_unique_m1 : int
        Number of targets that appear only in matrix1.
    num_unique_m2 : int
        Number of targets that appear only in matrix2.
    num_shared : int
        Number of shared targets that appear in both matrices.
    target_size_range : (int, int)
        Inclusive range for random target sizes (in number of pixels).
    partial_keep_ratio_range : (float, float)
        Range for the ratio of pixels kept from the base shape in partial overlap.
    partial_extra_cells_range : (int, int)
        Range for the number of extra pixels added around partial overlaps.
    seed : Optional[int]
        Random seed for reproducible generation (if provided).
    log_filename : str
        Name of the text file where generation details are logged.

    Returns
    -------
    Process_Data_Unit
        A PDU containing two tensors ("matrix1", "matrix2") and attributes
        describing the synthetic generation process.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        rng = random.Random(seed)
    else:
        rng = random.Random()

    rows, cols = matrix_shape
    m1 = np.zeros(matrix_shape, dtype=np.uint8)
    m2 = np.zeros(matrix_shape, dtype=np.uint8)

    occ1 = np.zeros_like(m1, dtype=bool)
    occ2 = np.zeros_like(m2, dtype=bool)

    log_lines: List[str] = []

    # ---------------------- Unique targets in matrix1 ---------------------- #
    for t in range(num_unique_m1):
        size = rng.randint(*target_size_range)
        coords = generate_connected_shape(matrix_shape, size, rng, occupied=occ1)
        for r, c in coords:
            m1[r, c] = 1
            occ1[r, c] = True
        log_lines.append(f"[M1-UNIQUE-{t+1}] size={len(coords)} coords={coords}")

    # ---------------------- Unique targets in matrix2 ---------------------- #
    for t in range(num_unique_m2):
        size = rng.randint(*target_size_range)
        coords = generate_connected_shape(matrix_shape, size, rng, occupied=occ2)
        for r, c in coords:
            m2[r, c] = 1
            occ2[r, c] = True
        log_lines.append(f"[M2-UNIQUE-{t+1}] size={len(coords)} coords={coords}")

    # -------------------------- Shared targets ----------------------------- #
    for t in range(num_shared):
        # Base shape in matrix1
        size_m1 = rng.randint(*target_size_range)
        base_coords = generate_connected_shape(matrix_shape, size_m1, rng, occupied=occ1)
        for r, c in base_coords:
            m1[r, c] = 1
            occ1[r, c] = True

        full_overlap = rng.choice([True, False])

        if full_overlap:
            # Copy the same shape to matrix2
            coords_m2 = list(base_coords)
            for r, c in coords_m2:
                m2[r, c] = 1
                occ2[r, c] = True
            log_lines.append(
                f"[SHARED-{t+1}] FULL size={len(base_coords)} "
                f"M1={base_coords} M2={coords_m2}"
            )
        else:
            # Build a partially overlapping shape in matrix2
            coords_m2 = _build_partial_overlap(
                base_coords,
                matrix_shape,
                rng,
                occ_m2=occ2,
                keep_ratio_range=partial_keep_ratio_range,
                extra_cells_range=partial_extra_cells_range,
            )
            for r, c in coords_m2:
                m2[r, c] = 1
                occ2[r, c] = True
            log_lines.append(
                f"[SHARED-{t+1}] PARTIAL M1_size={len(base_coords)} "
                f"M2_size={len(coords_m2)} M1={base_coords} M2={coords_m2}"
            )

    # -------------------------- Wrap as PDU ------------------------------- #
    tensor1 = ndarray_data(
        name="matrix1",
        data=m1,
        ndims=2,
        dims=[rows, cols],
        labels=["DP", "RG"],
        units="binary",
    )

    tensor2 = ndarray_data(
        name="matrix2",
        data=m2,
        ndims=2,
        dims=[rows, cols],
        labels=["DP", "RG"],
        units="binary",
    )

    attributes = base_data_attr(
        name="synthetic_rd_data",
        uuid=str(uuid.uuid4()),
        metadata={
            "description": "Synthetic Range–Doppler data with unique and shared targets",
            "matrix_shape": matrix_shape,
            "num_unique_m1": num_unique_m1,
            "num_unique_m2": num_unique_m2,
            "num_shared": num_shared,
            "target_size_range": target_size_range,
            "partial_keep_ratio_range": partial_keep_ratio_range,
            "partial_extra_cells_range": partial_extra_cells_range,
            "seed": seed,
        },
    )

    # Write log file for traceability
    log_path = os.path.join(os.getcwd(), log_filename)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("===== Synthetic Data Generation Log =====\n")
        f.write(f"matrix_shape={matrix_shape}\n")
        f.write(f"num_unique_m1={num_unique_m1}, num_unique_m2={num_unique_m2}, num_shared={num_shared}\n")
        f.write(f"target_size_range={target_size_range}\n")
        f.write(f"partial_keep_ratio_range={partial_keep_ratio_range}, "
                f"partial_extra_cells_range={partial_extra_cells_range}\n")
        f.write(f"seed={seed}\n")
        f.write("-----------------------------------------\n")
        for line in log_lines:
            f.write(line + "\n")
        f.write("=========================================\n")

    return Process_Data_Unit(
        attributes=attributes,
        tensors=[tensor1, tensor2],
        tables=None,
    )
