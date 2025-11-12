# generate_data.py
# -----------------------------------------------------------------------------
# Synthetic Range-Doppler data generator with controlled connected targets
# (4-connectivity only), unique + shared (full/partial overlap) targets,
# collision-avoidance, and detailed logging & metadata.
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import uuid
import random
from typing import List, Tuple, Optional, Dict

import numpy as np

# Project-local data wrappers (as per your environment)
# If these are your own dataclasses/modules, keep them. Otherwise you can swap
# to Python's @dataclass easily.
from procsess_data_unit import Process_Data_Unit, ndarray_data, base_data_attr


Coord = Tuple[int, int]
CoordList = List[Coord]


# ----------------------------- Helper Functions ----------------------------- #

def _neighbors4(r: int, c: int) -> List[Coord]:
    """Return 4-connected neighbor offsets around (r, c)."""
    return [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]


def _in_bounds(r: int, c: int, shape: Tuple[int, int]) -> bool:
    """Check if (r, c) lies within matrix bounds."""
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _add_if_free_and_new(coords: set[Coord],
                         frontier: List[Coord],
                         r: int,
                         c: int,
                         shape: Tuple[int, int],
                         occupied: Optional[np.ndarray] = None) -> bool:
    """
    Try adding (r, c) to the current connected set if:
      - in bounds
      - not already in coords
      - (optionally) not occupied in the global mask
    Returns True if added.
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


def generate_connected_shape(shape: Tuple[int, int],
                             size: int,
                             rng: random.Random,
                             occupied: Optional[np.ndarray] = None,
                             max_tries: int = 10_000) -> CoordList:
    """
    Generate a 4-connected shape (set of coordinates) of exactly `size` cells.
    If `occupied` is provided, generated cells must be False (free) in that mask.
    We use a simple randomized growth (like a 4-neighbor random walk / region grow).

    Raises ValueError if it fails to find a placement within `max_tries`.
    """
    rows, cols = shape
    tries = 0

    while tries < max_tries:
        tries += 1
        # Random seed cell (must be free if occupied mask is provided)
        sr = rng.randrange(rows)
        sc = rng.randrange(cols)
        if occupied is not None and occupied[sr, sc]:
            continue

        coords: set[Coord] = {(sr, sc)}
        frontier: List[Coord] = [(sr, sc)]

        # Grow until reaching size
        while len(coords) < size and frontier:
            # Pick a random frontier cell to expand from
            r, c = rng.choice(frontier)

            # Randomize the 4-neighbor order
            nbrs = _neighbors4(r, c)
            rng.shuffle(nbrs)

            progressed = False
            for nr, nc in nbrs:
                if _add_if_free_and_new(coords, frontier, nr, nc, shape, occupied):
                    progressed = True
                    if len(coords) >= size:
                        break

            # Remove dead-ends from frontier (no free 4-neighbors left)
            frontier = [
                (fr, fc) for (fr, fc) in frontier
                if any(
                    _in_bounds(fr + dr, fc + dc, shape) and
                    (fr + dr, fc + dc) not in coords and
                    (occupied is None or not occupied[fr + dr, fc + dc])
                    for (dr, dc) in [(1, 0), (-1, 0), (0, 1), (0, -1)]
                )
            ]

            # If we didn’t progress and all frontier is dead -> break
            if not progressed and not frontier:
                break

        if len(coords) == size:
            return list(coords)

        # else: restart with a new seed
    raise ValueError("Unable to generate a connected shape with the given constraints.")


def try_place_unique_target(matrix: np.ndarray,
                            occupied: np.ndarray,
                            size_range: Tuple[int, int],
                            rng: random.Random,
                            max_tries: int = 5_000) -> CoordList:
    """
    Place a single unique 4-connected target of random size within size_range onto `matrix`.
    Uses `occupied` to prevent unplanned collisions. Returns the coords if successful.
    Raises ValueError on failure.
    """
    rows, cols = matrix.shape
    min_sz, max_sz = size_range

    for _ in range(max_tries):
        sz = rng.randint(min_sz, max_sz)
        coords = generate_connected_shape((rows, cols), sz, rng, occupied=occupied)
        # Mark on matrix + occupied
        rr, cc = zip(*coords)
        matrix[np.array(rr), np.array(cc)] = 1
        occupied[np.array(rr), np.array(cc)] = True
        return coords

    raise ValueError("Failed to place a unique target after many attempts.")


def build_partial_overlap_for_shared(base_coords: CoordList,
                                     shape: Tuple[int, int],
                                     rng: random.Random,
                                     occupied_for_m2: np.ndarray,
                                     keep_ratio_range: Tuple[float, float] = (0.4, 0.8),
                                     extra_cells_range: Tuple[int, int] = (0, 5)) -> CoordList:
    """
    Build a connected set for Matrix-2 that partially overlaps with `base_coords` (Matrix-1).
    Strategy:
      1) Choose a random subset of base_coords to keep as overlap (ratio in keep_ratio_range).
      2) From a random kept cell (or a random overlap cell), grow additional 4-connected cells
         that are NOT in base_coords and not occupied in M2, to maintain connectivity.
      3) The final coords for M2 remain one connected component.

    NOTE: We respect 4-connectivity and do not introduce diagonal-only links.
    """
    base_set = set(base_coords)
    rows, cols = shape

    # 1) overlap subset
    keep_ratio = rng.uniform(*keep_ratio_range)
    keep_count = max(1, int(round(keep_ratio * len(base_coords))))
    overlap_subset = set(rng.sample(base_coords, keep_count))

    # 2) grow extra cells (if requested)
    extra_cells = rng.randint(*extra_cells_range)
    coords_m2: set[Coord] = set(overlap_subset)
    frontier = list(overlap_subset) if overlap_subset else [rng.choice(base_coords)]

    while len(coords_m2) < keep_count + extra_cells and frontier:
        r, c = rng.choice(frontier)
        nbrs = _neighbors4(r, c)
        rng.shuffle(nbrs)
        progressed = False
        for nr, nc in nbrs:
            # Must be inside bounds, not in base (to keep partial distinct cells),
            # not already in M2 coords, and not occupied in M2.
            if (_in_bounds(nr, nc, (rows, cols)) and
                (nr, nc) not in base_set and
                (nr, nc) not in coords_m2 and
                not occupied_for_m2[nr, nc]):
                coords_m2.add((nr, nc))
                frontier.append((nr, nc))
                progressed = True
                break
        # remove dead-ends
        frontier = [
            (fr, fc) for (fr, fc) in frontier
            if any(
                _in_bounds(fr + dr, fc + dc, (rows, cols)) and
                (fr + dr, fc + dc) not in coords_m2 and
                not occupied_for_m2[fr + dr, fc + dc]
                and (fr + dr, fc + dc) not in base_set
                for (dr, dc) in [(1, 0), (-1, 0), (0, 1), (0, -1)]
            )
        ]
        if not progressed and not frontier:
            break

    return list(coords_m2)


# ------------------------------- Main Generator ------------------------------ #

def generate_synthetic_rd_data(
    matrix_shape: Tuple[int, int] = (32, 1024),
    # By prompt: 6 unique in M1, 6 unique in M2, and 4 shared targets
    num_unique_m1: int = 6,
    num_unique_m2: int = 6,
    num_shared: int = 4,
    # Reasonable shape sizes for targets (inclusive)
    target_size_range: Tuple[int, int] = (3, 12),
    # Partial-overlap knobs
    partial_keep_ratio_range: Tuple[float, float] = (0.4, 0.8),
    partial_extra_cells_range: Tuple[int, int] = (0, 5),
    # Random seed for reproducibility
    seed: Optional[int] = None,
    # Logging
    log_filename: str = "synthetic_generation_log.txt",
) -> Process_Data_Unit:
    """
    Generate two synthetic Range-Doppler matrices with:
      - 6 unique targets in Matrix-1
      - 6 unique targets in Matrix-2
      - 4 shared targets (each present in both matrices).
        * Some shared targets fully overlap.
        * Some shared targets partially overlap (still 4-connected).
    Constraints:
      * Each target is 4-connected (no diagonal-only links).
      * Unique targets avoid collisions with existing ones in the same matrix.
      * Shared targets are explicitly constructed (full or partial).
      * All coordinates are logged to a text file for traceability.

    Returns:
      Process_Data_Unit with two tensors named "matrix1"/"matrix2" and metadata that
      includes detailed target coordinate listings.
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

    # Occupancy masks to avoid unplanned collisions per matrix
    occ1 = np.zeros_like(m1, dtype=bool)
    occ2 = np.zeros_like(m2, dtype=bool)

    # Keep logs + metadata
    log_lines: List[str] = []
    m1_unique_targets: List[CoordList] = []
    m2_unique_targets: List[CoordList] = []
    shared_targets: List[Dict[str, object]] = []  # [{"type": "full"/"partial", "m1": [...], "m2": [...]}]

    # ------------------------ Place unique targets (M1) ----------------------- #
    for t in range(num_unique_m1):
        coords = try_place_unique_target(
            m1, occ1, size_range=target_size_range, rng=rng
        )
        m1_unique_targets.append(coords)
        log_lines.append(f"[M1-UNIQUE-{t+1}] size={len(coords)} coords={coords}")

    # ------------------------ Place unique targets (M2) ----------------------- #
    for t in range(num_unique_m2):
        coords = try_place_unique_target(
            m2, occ2, size_range=target_size_range, rng=rng
        )
        m2_unique_targets.append(coords)
        log_lines.append(f"[M2-UNIQUE-{t+1}] size={len(coords)} coords={coords}")

    # -------------------------- Place shared targets -------------------------- #
    for s in range(num_shared):
        # 1) Build the base (Matrix-1) shape
        size_m1 = rng.randint(*target_size_range)
        base_coords = generate_connected_shape(matrix_shape, size_m1, rng, occupied=occ1)
        rr, cc = zip(*base_coords)
        m1[np.array(rr), np.array(cc)] = 1
        occ1[np.array(rr), np.array(cc)] = True

        # Randomly decide full vs partial overlap
        full_overlap = rng.choice([True, False])

        if full_overlap:
            coords_m2 = list(base_coords)
            # Ensure they do not collide with something already in M2 beyond intended overlap
            # If any cell is already taken in M2, we can attempt to place the same shape elsewhere
            # but the prompt allows full overlap for the same target, so placing directly is correct.
            rr2, cc2 = zip(*coords_m2)
            m2[np.array(rr2), np.array(cc2)] = 1
            occ2[np.array(rr2), np.array(cc2)] = True

            shared_targets.append({
                "type": "full",
                "m1": list(base_coords),
                "m2": list(coords_m2),
            })
            log_lines.append(f"[SHARED-{s+1}] FULL size={len(base_coords)} M1={base_coords} M2={coords_m2}")
        else:
            # Build partial overlap for M2: keep a subset of base and add extra connected cells
            coords_m2 = build_partial_overlap_for_shared(
                base_coords,
                matrix_shape,
                rng,
                occupied_for_m2=occ2,
                keep_ratio_range=partial_keep_ratio_range,
                extra_cells_range=partial_extra_cells_range,
            )
            # Mark on M2
            rr2, cc2 = zip(*coords_m2)
            m2[np.array(rr2), np.array(cc2)] = 1
            occ2[np.array(rr2), np.array(cc2)] = True

            shared_targets.append({
                "type": "partial",
                "m1": list(base_coords),
                "m2": list(coords_m2),
            })
            log_lines.append(f"[SHARED-{s+1}] PARTIAL M1_size={len(base_coords)} M2_size={len(coords_m2)} "
                             f"M1={base_coords} M2={coords_m2}")

    # ----------------------------- Wrap as tensors ---------------------------- #
    tensor1 = ndarray_data(
        name="matrix1",
        data=m1,
        ndims=2,
        dims=[rows, cols],
        labels=["DP", "RG"],   # Doppler, Range (keep your preferred convention)
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
            "description": "Synthetic Range-Doppler data with unique and shared (full/partial) targets",
            "matrix_shape": matrix_shape,
            "num_unique_m1": num_unique_m1,
            "num_unique_m2": num_unique_m2,
            "num_shared": num_shared,
            "target_size_range": target_size_range,
            "partial_keep_ratio_range": partial_keep_ratio_range,
            "partial_extra_cells_range": partial_extra_cells_range,
            "seed": seed,
            "targets": {
                "m1_unique": [list(coords) for coords in m1_unique_targets],
                "m2_unique": [list(coords) for coords in m2_unique_targets],
                "shared": shared_targets,
            },
        },
    )

    # ------------------------------- Write log -------------------------------- #
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

    # ------------------------------- Return PDU ------------------------------- #
    return Process_Data_Unit(
        attributes=attributes,
        tensors=[tensor1, tensor2],
        tables=None,
    )
