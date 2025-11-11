from procsess_data_unit import dataframe_data
from typing import List
import numpy as np
import pandas as pd
from scipy.ndimage import label


def compare_target_matrices(pdu, connectivity=1):
    """
    Compare targets between two matrices in a Process_Data_Unit.
    Each target is defined as a connected component (4-connectivity by default).
    
    Returns:
        Process_Data_Unit with 3 tables added:
        - matrix1_only
        - matrix2_only
        - overlap
    """

    # === Get matrices ===
    matrix1 = pdu.tensors[0].data
    matrix2 = pdu.tensors[1].data

    if matrix1.shape != matrix2.shape:
        raise ValueError(f"Matrices have different shapes: {matrix1.shape} vs {matrix2.shape}")

    # === Label connected components ===
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]]) if connectivity == 1 else None  # 4-connectivity

    labeled1, num1 = label(matrix1, structure=structure)
    labeled2, num2 = label(matrix2, structure=structure)

    # === Extract coordinates per target ===
    targets1 = {i: np.argwhere(labeled1 == i).tolist() for i in range(1, num1 + 1)}
    targets2 = {i: np.argwhere(labeled2 == i).tolist() for i in range(1, num2 + 1)}

    # === Prepare records ===
    matrix1_only_records = []
    matrix2_only_records = []
    overlap_records = []
    used_targets2 = set()

    # === Compare each target from matrix1 to all in matrix2 ===
    for id1, coords1 in targets1.items():
        coords1_set = set(map(tuple, coords1))
        found_overlap = False

        for id2, coords2 in targets2.items():
            coords2_set = set(map(tuple, coords2))
            intersection = coords1_set & coords2_set

            if intersection:
                found_overlap = True
                used_targets2.add(id2)

                overlap_percent = (
                    len(intersection) / max(len(coords1_set), len(coords2_set)) * 100
                )

                overlap_records.append({
                    "target_id_matrix1": f"M1_{id1}",
                    "target_id_matrix2": f"M2_{id2}",
                    "rg_indices": [[r[1] for r in coords1], [r[1] for r in coords2]],
                    "dp_indices": [[r[0] for r in coords1], [r[0] for r in coords2]],
                    "overlap_percent": overlap_percent
                })

        # No overlaps found → target exists only in matrix1
        if not found_overlap:
            matrix1_only_records.append({
                "target_id": f"M1_{id1}",
                "rg_indices": [r[1] for r in coords1],
                "dp_indices": [r[0] for r in coords1]
            })

    # === Find targets from matrix2 not used in any overlap ===
    for id2, coords2 in targets2.items():
        if id2 not in used_targets2:
            matrix2_only_records.append({
                "target_id": f"M2_{id2}",
                "rg_indices": [r[1] for r in coords2],
                "dp_indices": [r[0] for r in coords2]
            })

    # === Convert to DataFrames ===
    df_matrix1_only = pd.DataFrame(matrix1_only_records)
    df_matrix2_only = pd.DataFrame(matrix2_only_records)
    df_overlap = pd.DataFrame(overlap_records)

    # === Add tables safely to Process_Data_Unit ===
    if pdu.tables is None:
        pdu.tables = []

    pdu.tables.extend([
        dataframe_data(name="matrix1_only", data=df_matrix1_only),
        dataframe_data(name="matrix2_only", data=df_matrix2_only),
        dataframe_data(name="overlap", data=df_overlap)
    ])

    return pdu
