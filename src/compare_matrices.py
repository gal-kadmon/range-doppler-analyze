# compare_matrices.py
# -----------------------------------------------------------------------------
# Range–Doppler Target Comparison Module
# -----------------------------------------------------------------------------
# This module compares two binary Range–Doppler matrices stored inside a
# Process_Data_Unit (PDU). A “target” is defined as a connected component under
# 8-connectivity (i.e., each pixel is connected to its horizontal, vertical,
# and diagonal neighbors).
#
# The module identifies:
#   • Targets unique to matrix 1
#   • Targets unique to matrix 2
#   • Targets present in both matrices (full or partial spatial overlap)
#
# Overlap is quantified using the Jaccard index:
#     J(A, B) = |A ∩ B| / |A ∪ B|
#
# -----------------------------------------------------------------------------
# Output — updated PDU content
# -----------------------------------------------------------------------------
# The function returns the updated Process_Data_Unit (PDU) object with:
#   • 1. Three pandas DataFrames that are added to pdu.tables:
#             
#        • matrix1_only / matrix2_only:
#             Columns:
#                 - "target_id"  : string identifier ("M1_X" or "M2_X")
#                 - "coords"     : list of (range, doppler) index pairs
#                 - "size"       : number of pixels in the target
#
#        • overlap (contains full or partial overlap):
#             Columns:
#                 - "target_id_matrix1" : ID of matrix1 target ("M1_X")
#                 - "target_id_matrix2" : ID of matrix2 target ("M2_X")
#                 - "coords_matrix1"    : list of (range, doppler) pixels of the
#                                         matching target in matrix1
#                 - "coords_matrix2"    : list of (range, doppler) pixels of the
#                                         matching target in matrix2
#                 - "overlap_cells"     : number of pixels shared by both targets
#                 - "overlap_percent"   : Jaccard similarity (0–100%)
#
#   • 2. Per-pixel metadata is added to the original tensors: 
#        a tensor_extra_attr entry is appended to the corresponding tensor’s metadata list.
#
#        • Each entry contains:
#           - x, y                 : pixel coordinates (range, doppler)
#           - metadata dict with:
#                 "target_id"      : for unique targets
#                 "target_pair"    : for overlapping targets (pair of IDs)
#                 "type"           : classification:
#                                      "matrix1_only"
#                                      "matrix2_only"
#                                      "overlap"
#
#   • 3. A third tensor "targets_mask" is appended to pdu.tensors:
#        • It is a 2D numpy array categorical matrix with the same shape as the original matrices.
#        • Each cell encodes the combined target classification from both matrices:
#
#          0 = empty
#          1 = matrix1-only target pixel
#          2 = matrix2-only target pixel
#          3 = pixel belonging to overlapping target pair
#
#   • 4. Legend describing the meaning of mask categories:
#        • The legend is inserted into the global PDU attributes:
#
#           pdu.attributes.metadata["targets_mask_legend"] = {
#               0: "empty",
#               1: "matrix1_only",
#               2: "matrix2_only",
#               3: "overlap"
#           }
#   
# -----------------------------------------------------------------------------
# Summary:
#   After running compare_target_matrices(pdu), the same PDU becomes a complete,
#   self-contained description of all detected targets, their pixel-level
#   membership, and their spatial relationships across both matrices.
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
from scipy.ndimage import label

from process_data_unit import (
    dataframe_data,
    ndarray_data,
    tensor_extra_attr,
)


def compare_target_matrices(pdu):
    """
    Compare two binary Range–Doppler matrices inside a Process_Data_Unit (PDU).

    The PDU must contain exactly two tensors:
        pdu.tensors[0] → matrix1 (2D array)
        pdu.tensors[1] → matrix2 (same shape)

    Targets are extracted using 8-connectivity. The function identifies:
        • Targets unique to matrix1
        • Targets unique to matrix2
        • Target pairs that spatially overlap (via Jaccard similarity)

    The same PDU object is returned, enriched with:
        • Three tables:
              - matrix1_only  : target_id, coords, size
              - matrix2_only  : target_id, coords, size
              - overlap       : target_id_matrix1, target_id_matrix2,
                                coords_matrix1, coords_matrix2,
                                overlap_cells, overlap_percent
        • Per-pixel metadata added to matrix1/matrix2 tensors,
          describing each target pixel and its classification.
        • A new categorical tensor "targets_mask" (0/1/2/3) indicating:
              0=empty, 1=matrix1_only, 2=matrix2_only, 3=overlap
        • A legend stored in pdu.attributes.metadata["targets_mask_legend"].

    Returns the SAME PDU instance, updated with all comparison outputs.
    """

    # Retrieve matrices
    matrix1 = pdu.tensors[0].data
    matrix2 = pdu.tensors[1].data

    if matrix1.shape != matrix2.shape:
        raise ValueError("Both matrices must share the same shape.")

    shape = matrix1.shape

    # Connected-components labeling: 8 possible neighbors
    structure = np.ones((3, 3), dtype=int)

    labeled1, num1 = label(matrix1, structure=structure)
    labeled2, num2 = label(matrix2, structure=structure)

    def extract_targets(labeled, count):
        """Return dict: target_id → array of (range, doppler) coordinates."""
        return {i: np.argwhere(labeled == i) for i in range(1, count + 1)}

    targets1 = extract_targets(labeled1, num1)
    targets2 = extract_targets(labeled2, num2)

    # Prepare table record lists
    m1_only_records = []
    m2_only_records = []
    overlap_records = []
    used_targets2 = set()

    # Bounding box computation for efficient overlap pruning
    def bounding_box(coords):
        """Return the minimal (r_min, r_max, c_min, c_max) rectangle enclosing all target pixels."""
        r = coords[:, 0]
        c = coords[:, 1]
        return r.min(), r.max(), c.min(), c.max()

    # --------------------------------------------------------------------------
    # Compare every target in matrix1 against every target in matrix2.
    # A fast bounding-box check rejects pairs that cannot overlap; 
    # for the remaining candidates, pixel-level intersection is computed to determine
    # whether targets are unique or overlapping.
    # --------------------------------------------------------------------------
    
    # dict: target_id → (r_min, r_max, c_min, c_max) bounding-box tuple for matrix2
    boxes2 = {i: bounding_box(coords) for i, coords in targets2.items()}

    # Compare all targets (matrix1 vs matrix2)
    for id1, coords1 in targets1.items():
        coords1_set = set(map(tuple, coords1.tolist()))
        rmin1, rmax1, cmin1, cmax1 = bounding_box(coords1)

        found_overlap = False

        for id2, coords2 in targets2.items():
            # bounding-box rejection
            rmin2, rmax2, cmin2, cmax2 = boxes2[id2]

            if (rmax1 < rmin2 or rmax2 < rmin1 or
                cmax1 < cmin2 or cmax2 < cmin1):
                continue

            coords2_set = set(map(tuple, coords2.tolist()))
            intersection = coords1_set & coords2_set
            if not intersection:
                continue

            # Overlapping targets detected - targets id1, id2 have at least some intersection
            found_overlap = True
            used_targets2.add(id2)

            overlap_percent = (
                len(intersection) / len(coords1_set | coords2_set) * 100
            )

            coords1_list = list(coords1_set)
            coords2_list = list(coords2_set)

            overlap_records.append({
                "target_id_matrix1": f"M1_{id1}",
                "target_id_matrix2": f"M2_{id2}",
                "coords_matrix1": coords1_list,
                "coords_matrix2": coords2_list,
                "overlap_cells": len(intersection),
                "overlap_percent": overlap_percent,
            })

        # No overlap - target id1 apppends to m1_only_records
        if not found_overlap:
            coords_list = list(coords1_set)
            m1_only_records.append({
                "target_id": f"M1_{id1}",
                "coords": coords_list,
                "size": len(coords_list),
            })

    # Targets unique to matrix2 - targets that were not added to overlap_records
    for id2, coords2 in targets2.items():
        if id2 not in used_targets2:
            coords2_set = set(map(tuple, coords2.tolist()))
            coords_list = list(coords2_set)

            m2_only_records.append({
                "target_id": f"M2_{id2}",
                "coords": coords_list,
                "size": len(coords_list),
            })

    # Convert record lists to DataFrames
    df_m1_only = pd.DataFrame(m1_only_records)
    df_m2_only = pd.DataFrame(m2_only_records)
    df_overlap = pd.DataFrame(overlap_records)

    if not df_m1_only.empty:
        df_m1_only = df_m1_only[["target_id", "coords", "size"]]

    if not df_m2_only.empty:
        df_m2_only = df_m2_only[["target_id", "coords", "size"]]

    if not df_overlap.empty:
        df_overlap = df_overlap[
            [
                "target_id_matrix1",
                "target_id_matrix2",
                "coords_matrix1",
                "coords_matrix2",
                "overlap_cells",
                "overlap_percent",
            ]
        ]

    # Build per-pixel metadata for each target cell in each original tensor
    tensor1_metadata = []
    tensor2_metadata = []

    # matrix1-only
    for _, row in df_m1_only.iterrows():
        for r, c in row["coords"]:
            tensor1_metadata.append(
                tensor_extra_attr(
                    x=int(r),
                    y=int(c),
                    metadata={
                        "target_id": row["target_id"],
                        "type": "matrix1_only",
                    }
                )
            )

    # matrix2-only
    for _, row in df_m2_only.iterrows():
        for r, c in row["coords"]:
            tensor2_metadata.append(
                tensor_extra_attr(
                    x=int(r),
                    y=int(c),
                    metadata={
                        "target_id": row["target_id"],
                        "type": "matrix2_only",
                    }
                )
            )

    # overlapping
    for _, row in df_overlap.iterrows():
        # matrix1 pixels
        for r, c in row["coords_matrix1"]:
            tensor1_metadata.append(
                tensor_extra_attr(
                    x=int(r),
                    y=int(c),
                    metadata={
                        "target_pair": (
                            row["target_id_matrix1"],
                            row["target_id_matrix2"]
                        ),
                        "type": "overlap",
                    }
                )
            )
        # matrix2 pixels
        for r, c in row["coords_matrix2"]:
            tensor2_metadata.append(
                tensor_extra_attr(
                    x=int(r),
                    y=int(c),
                    metadata={
                        "target_pair": (
                            row["target_id_matrix1"],
                            row["target_id_matrix2"]
                        ),
                        "type": "overlap",
                    }
                )
            )

    pdu.tensors[0].metadata = tensor1_metadata
    pdu.tensors[1].metadata = tensor2_metadata

    # Build categorical mask tensor: 0/1/2/3
    mask = np.zeros(shape, dtype=np.uint8)

    # matrix1-only
    for _, row in df_m1_only.iterrows():
        for r, c in row["coords"]:
            mask[r, c] = 1

    # matrix2-only
    for _, row in df_m2_only.iterrows():
        for r, c in row["coords"]:
            mask[r, c] = 2

    # overlapping (both target regions)
    for _, row in df_overlap.iterrows():
        for r, c in row["coords_matrix1"]:
            mask[r, c] = 3
        for r, c in row["coords_matrix2"]:
            mask[r, c] = 3

    mask_tensor = ndarray_data(
        name="targets_mask",
        data=mask,
        ndims=2,
        dims=list(mask.shape),
        labels=["DP", "RG"],
        units="categorical",
        metadata=None
    )

    pdu.tensors.append(mask_tensor)
    if pdu.attributes.metadata is None:
      pdu.attributes.metadata = {}

    # Add new key to metadata dict
    pdu.attributes.metadata["targets_mask_legend"] = {
        0: "empty",
        1: "matrix1_only",
        2: "matrix2_only",
        3: "overlap"
    }

    # Insert tables into PDU
    if pdu.tables is None:
        pdu.tables = []

    pdu.tables.extend([
        dataframe_data(name="matrix1_only", data=df_m1_only),
        dataframe_data(name="matrix2_only", data=df_m2_only),
        dataframe_data(name="overlap", data=df_overlap),
    ])

    return pdu