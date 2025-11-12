# visualize_comparison.py
# -----------------------------------------------------------------------------
# Visualize comparison of Range-Doppler targets:
# - Red:  Matrix1-only
# - Blue: Matrix2-only
# - Purple: Full overlap or partial overlap cells
# Dynamic scaling (zoom-in) to non-empty areas only.
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import numpy as np
import pandas as pd


def plot_targets(df_m1_only: pd.DataFrame,
                 df_m2_only: pd.DataFrame,
                 df_overlap: pd.DataFrame,
                 matrix_shape: tuple[int, int]):
    """
    Display targets comparison on a dynamic grid.
    - Red   → Matrix1-only
    - Blue  → Matrix2-only
    - Purple → Overlap (full or partial)
    """
    # ----------------------------- Helper Functions ----------------------------- #
    def extract_all_coords(df: pd.DataFrame, coord_field="coords"):
        """Extract list of (r, c) from a dataframe field."""
        if df.empty:
            return []
        if coord_field in df.columns:
            coords = [tuple(x) for row in df[coord_field] for x in row]
        elif {"dp_indices", "rg_indices"} <= set(df.columns):
            coords = []
            for _, row in df.iterrows():
                dp = row["dp_indices"]
                rg = row["rg_indices"]
                if isinstance(dp, list) and all(isinstance(x, int) for x in dp):
                    coords.extend(zip(dp, rg))
                else:
                    for dp_sub, rg_sub in zip(dp, rg):
                        coords.extend(zip(dp_sub, rg_sub))
        else:
            coords = []
        return list(set(coords))

    def build_rectangles(coords, color):
        """Return a list of matplotlib Rectangles from coordinates."""
        return [Rectangle((col_scale[c], row_scale[r]), 1, 1, facecolor=color, edgecolor='gray')
                for r, c in coords if r in row_scale and c in col_scale]

    # -------------------------- Gather all coordinates -------------------------- #
    coords_m1 = extract_all_coords(df_m1_only)
    coords_m2 = extract_all_coords(df_m2_only)
    coords_overlap = []

    if not df_overlap.empty:
        if "overlap_coords" in df_overlap.columns:
            for row in df_overlap["overlap_coords"]:
                coords_overlap.extend(row)
        else:
            # Fallback to old format
            for _, row in df_overlap.iterrows():
                dp_lists = row["dp_indices"]
                rg_lists = row["rg_indices"]
                for dp_sub, rg_sub in zip(dp_lists, rg_lists):
                    coords_overlap.extend(zip(dp_sub, rg_sub))

    all_coords = coords_m1 + coords_m2 + coords_overlap
    if not all_coords:
        print("No targets to visualize.")
        return

    all_coords = np.array(all_coords)
    unique_rows = np.unique(all_coords[:, 0])
    unique_cols = np.unique(all_coords[:, 1])

    # Dynamic scaling maps
    row_scale = {val: i for i, val in enumerate(sorted(unique_rows))}
    col_scale = {val: i for i, val in enumerate(sorted(unique_cols))}

    # -------------------------- Build all patches -------------------------- #
    patches = []
    patches.extend(build_rectangles(coords_m1, 'red'))
    patches.extend(build_rectangles(coords_m2, 'blue'))
    patches.extend(build_rectangles(coords_overlap, 'purple'))

    # -------------------------- Create the plot --------------------------- #
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_aspect('equal')
    ax.set_facecolor('white')

    # Add all rectangles as a PatchCollection (faster)
    collection = PatchCollection(patches, match_original=True)
    ax.add_collection(collection)

    # Configure gridlines and ticks
    ax.set_xticks(range(len(unique_cols) + 1))
    ax.set_yticks(range(len(unique_rows) + 1))
    ax.grid(True, color='gray', linewidth=0.4, alpha=0.5)
    ax.invert_yaxis()

    # Set plot limits to visible region only
    ax.set_xlim(0, len(unique_cols))
    ax.set_ylim(0, len(unique_rows))

    # Add legend
    handles = [
        Rectangle((0, 0), 1, 1, color='red', label='Matrix1 Only'),
        Rectangle((0, 0), 1, 1, color='blue', label='Matrix2 Only'),
        Rectangle((0, 0), 1, 1, color='purple', label='Overlap'),
    ]
    ax.legend(handles=handles, loc='upper right')

    # Labels and title
    ax.set_title("Targets Comparison (Dynamic Grid Zoomed)", fontsize=14)
    ax.set_xlabel("Range Index (scaled)")
    ax.set_ylabel("Doppler Index (scaled)")

    plt.tight_layout()
    plt.show()
