# visualize_comparison.py
# -----------------------------------------------------------------------------
# Visualization module for Range–Doppler target comparison.
#
# This tool visualizes the comparison result stored in three tables:
#   • matrix1_only:  target_id, coords, size
#   • matrix2_only:  target_id, coords, size
#   • overlap:       target_id_matrix1, target_id_matrix2,
#                    coords_matrix1, coords_matrix2,
#                    overlap_cells, overlap_percent
#
# Each coordinate is expressed as (range_index, doppler_index).
#
# The visualization:
#   • Draws all target pixels with color coding:
#         red     → pixel from matrix1-only target
#         blue    → pixel from matrix2-only target
#         purple  → pixel that belongs to overlapping targets
#
#   • Compresses/zooms the grid to show only coordinate rows/columns
#     that contain targets, but labels axes with ORIGINAL RD indices.
#
# This yields a compact visualization that preserves the physical
# meaning of indices while focusing only on target regions.
# -----------------------------------------------------------------------------

from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection


def plot_targets(
    df_m1_only: pd.DataFrame,
    df_m2_only: pd.DataFrame,
    df_overlap: pd.DataFrame,
    matrix_shape: Tuple[int, int],
) -> None:
    """
    Visualize comparison results between two Range–Doppler matrices.

    Parameters
    ----------
    df_m1_only : DataFrame
        Contains columns ["target_id", "coords", "size"],
        where coords is a list of (r,c) index pairs.

    df_m2_only : DataFrame
        Same format as df_m1_only.

    df_overlap : DataFrame
        Contains columns:
            "target_id_matrix1",
            "target_id_matrix2",
            "coords_matrix1",
            "coords_matrix2",
            "overlap_cells",
            "overlap_percent"

    matrix_shape : tuple(int,int)
        Shape of the original matrices (range, doppler).
        Used only as reference for axis labeling.
    """

    # ----------------------------------------------------------------------
    # Collect all coordinates from all tables
    # ----------------------------------------------------------------------
    all_coords = []

    if not df_m1_only.empty:
        for _, row in df_m1_only.iterrows():
            all_coords.extend(row["coords"])

    if not df_m2_only.empty:
        for _, row in df_m2_only.iterrows():
            all_coords.extend(row["coords"])

    if not df_overlap.empty:
        for _, row in df_overlap.iterrows():
            all_coords.extend(row["coords_matrix1"])
            all_coords.extend(row["coords_matrix2"])

    if not all_coords:
        print("No targets to visualize.")
        return

    all_coords_arr = np.array(all_coords, dtype=int)
    unique_rows = np.unique(all_coords_arr[:, 0])
    unique_cols = np.unique(all_coords_arr[:, 1])

    # ----------------------------------------------------------------------
    # Build compressed-to-original mapping for zoomed view
    # ----------------------------------------------------------------------
    row_map = {r: i for i, r in enumerate(sorted(unique_rows))}
    col_map = {c: i for i, c in enumerate(sorted(unique_cols))}

    # ----------------------------------------------------------------------
    # Utility to build colored pixel rectangles
    # ----------------------------------------------------------------------
    def build_rectangles(coords, color):
        patches = []
        for (r, c) in coords:
            if r not in row_map or c not in col_map:
                continue
            rr = row_map[r]
            cc = col_map[c]
            patches.append(
                Rectangle((cc, rr), 1, 1, facecolor=color, edgecolor="gray")
            )
        return patches

    # ----------------------------------------------------------------------
    # Collect patches for each category
    # ----------------------------------------------------------------------
    patches_m1_only = []
    patches_m2_only = []
    patches_overlap = []

    # Matrix1-only: red
    if not df_m1_only.empty:
        for _, row in df_m1_only.iterrows():
            patches_m1_only.extend(build_rectangles(row["coords"], "red"))

    # Matrix2-only: blue
    if not df_m2_only.empty:
        for _, row in df_m2_only.iterrows():
            patches_m2_only.extend(build_rectangles(row["coords"], "blue"))

    # Overlapping targets: purple (full or partial)
    if not df_overlap.empty:
        for _, row in df_overlap.iterrows():

            coords1 = set(row["coords_matrix1"])
            coords2 = set(row["coords_matrix2"])
            intersection = coords1 & coords2

            full_overlap = (coords1 == coords2 == intersection)

            if full_overlap:
                patches_overlap.extend(build_rectangles(coords1, "purple"))

            else:
                # Partial overlap: separate regions
                non1 = coords1 - intersection
                non2 = coords2 - intersection

                patches_m1_only.extend(build_rectangles(non1, "red"))
                patches_m2_only.extend(build_rectangles(non2, "blue"))
                patches_overlap.extend(build_rectangles(intersection, "purple"))

    # ----------------------------------------------------------------------
    # Create the figure
    # ----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_aspect("equal")
    ax.set_facecolor("white")

    # Add patch collections
    if patches_m1_only:
        ax.add_collection(PatchCollection(patches_m1_only, match_original=True))
    if patches_m2_only:
        ax.add_collection(PatchCollection(patches_m2_only, match_original=True))
    if patches_overlap:
        ax.add_collection(PatchCollection(patches_overlap, match_original=True))

    # ----------------------------------------------------------------------
    # Configure axes: compressed grid but with ORIGINAL index labels
    # ----------------------------------------------------------------------
    ax.set_xticks(range(len(unique_cols)))
    ax.set_xticklabels(sorted(unique_cols), rotation=45)

    ax.set_yticks(range(len(unique_rows)))
    ax.set_yticklabels(sorted(unique_rows))

    ax.grid(True, color="gray", linewidth=0.4, alpha=0.6)
    ax.invert_yaxis()

    ax.set_xlim(0, len(unique_cols))
    ax.set_ylim(0, len(unique_rows))

    # ----------------------------------------------------------------------
    # Legend
    # ----------------------------------------------------------------------
    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor="red", edgecolor="gray", label="Matrix1 Only"),
        Rectangle((0, 0), 1, 1, facecolor="blue", edgecolor="gray", label="Matrix2 Only"),
        Rectangle((0, 0), 1, 1, facecolor="purple", edgecolor="gray", label="Overlap"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    # ----------------------------------------------------------------------
    # Titles
    # ----------------------------------------------------------------------
    ax.set_title("Range–Doppler Target Comparison (Zoomed with Original Axis Labels)")
    ax.set_xlabel("Range Index")
    ax.set_ylabel("Doppler Index")

    plt.tight_layout()
    plt.show()
