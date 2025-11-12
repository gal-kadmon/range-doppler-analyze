# compare_matrices.py
# -----------------------------------------------------------------------------
# Compare two Range-Doppler matrices, identify unique and overlapping targets.
# -----------------------------------------------------------------------------

from typing import List, Set, Tuple
import numpy as np
import pandas as pd
from scipy.ndimage import label
from procsess_data_unit import dataframe_data


def compare_target_matrices(pdu, connectivity: int = 1):
    """
    Compare labeled targets between two Range-Doppler matrices.
    Each target is a connected component (default 4-connectivity).
    Returns the same Process_Data_Unit with new DataFrames:
      - matrix1_only
      - matrix2_only
      - overlap
    """
    # --- Retrieve matrices ---
    matrix1 = pdu.tensors[0].data
    matrix2 = pdu.tensors[1].data

    if matrix1.shape != matrix2.shape:
        raise ValueError(f"Matrices have different shapes: {matrix1.shape} vs {matrix2.shape}")

    # --- Connectivity structure ---
    if connectivity == 1:  # 4-connectivity
        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]], dtype=int)
    elif connectivity == 2:  # 8-connectivity
        structure = np.ones((3, 3), dtype=int)
    else:
        raise ValueError("connectivity must be 1 (4-neighbor) or 2 (8-neighbor)")

    # --- Label components ---
    labeled1, num1 = label(matrix1, structure=structure)
    labeled2, num2 = label(matrix2, structure=structure)

    # --- Extract coordinates for each target ---
    def extract_targets(labeled, count):
        return {i: np.argwhere(labeled == i) for i in range(1, count + 1)}

    targets1 = extract_targets(labeled1, num1)
    targets2 = extract_targets(labeled2, num2)

    # --- Prepare result containers ---
    matrix1_only_records = []
    matrix2_only_records = []
    overlap_records = []
    used_targets2: Set[int] = set()

    # --- Spatial index optimization ---
    # build dict of bounding boxes for M2 to prune comparisons
    def bounding_box(coords):
        rows = coords[:, 0]
        cols = coords[:, 1]
        return rows.min(), rows.max(), cols.min(), cols.max()

    boxes2 = {i: bounding_box(coords) for i, coords in targets2.items()}

    # --- Compare targets efficiently ---
    for id1, coords1 in targets1.items():
        rmin1, rmax1, cmin1, cmax1 = bounding_box(coords1)
        coords1_set = set(map(tuple, coords1.tolist()))
        found_overlap = False

        for id2, coords2 in targets2.items():
            # Skip if bounding boxes do not intersect at all
            rmin2, rmax2, cmin2, cmax2 = boxes2[id2]
            if rmax1 < rmin2 or rmax2 < rmin1 or cmax1 < cmin2 or cmax2 < cmin1:
                continue

            coords2_set = set(map(tuple, coords2.tolist()))
            intersection = coords1_set & coords2_set
            if not intersection:
                continue

            found_overlap = True
            used_targets2.add(id2)

            # Use symmetric overlap metric (Jaccard)
            overlap_percent = len(intersection) / len(coords1_set | coords2_set) * 100

            overlap_records.append({
                "target_id_matrix1": f"M1_{id1}",
                "target_id_matrix2": f"M2_{id2}",
                "size_matrix1": len(coords1_set),
                "size_matrix2": len(coords2_set),
                "overlap_cells": len(intersection),
                "overlap_percent": overlap_percent,
                "overlap_coords": list(intersection),
            })

        if not found_overlap:
            matrix1_only_records.append({
                "target_id": f"M1_{id1}",
                "size": len(coords1_set),
                "coords": list(coords1_set),
            })

    # --- Find matrix2-only targets ---
    for id2, coords2 in targets2.items():
        if id2 not in used_targets2:
            coords2_set = set(map(tuple, coords2.tolist()))
            matrix2_only_records.append({
                "target_id": f"M2_{id2}",
                "size": len(coords2_set),
                "coords": list(coords2_set),
            })

    # --- Convert to DataFrames ---
    df_m1_only = pd.DataFrame(matrix1_only_records)
    df_m2_only = pd.DataFrame(matrix2_only_records)
    df_overlap = pd.DataFrame(overlap_records)

    # --- Update Process_Data_Unit ---
    if pdu.tables is None:
        pdu.tables = []
    pdu.tables.extend([
        dataframe_data(name="matrix1_only", data=df_m1_only),
        dataframe_data(name="matrix2_only", data=df_m2_only),
        dataframe_data(name="overlap", data=df_overlap),
    ])

    return pdu
