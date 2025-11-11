import matplotlib.pyplot as plt
import numpy as np
import  pandas as pd

def plot_targets(matrix1_only, matrix2_only, overlap, shape):
    """
    Visualize targets with dynamic grid scaling:
    - Cells with targets are visually larger
    - Empty cells are shrunk
    """

    # Combine all target coordinates
    def all_coords(df):
        coords = []
        for _, row in df.iterrows():
            dp_lists = row['dp_indices']
            rg_lists = row['rg_indices']
            # Ensure lists of lists
            if all(isinstance(x, int) for x in dp_lists):
                dp_lists = [dp_lists]
                rg_lists = [rg_lists]
            for dp_sub, rg_sub in zip(dp_lists, rg_lists):
                coords.extend(zip(dp_sub, rg_sub))
        return coords

    coords_m1 = all_coords(matrix1_only)
    coords_m2 = all_coords(matrix2_only)
    coords_overlap = []
    for _, row in overlap.iterrows():
        dp_lists = row['dp_indices']
        rg_lists = row['rg_indices']
        for dp_sub, rg_sub in zip(dp_lists, rg_lists):
            coords_overlap.extend(zip(dp_sub, rg_sub))

    all_coords_total = coords_m1 + coords_m2 + coords_overlap
    all_coords_total = np.array(all_coords_total)

    # Calculate unique sorted positions
    unique_rows = np.unique(all_coords_total[:, 0])
    unique_cols = np.unique(all_coords_total[:, 1])

    # Create mapping from original index to scaled coordinate
    row_scale = {val: i for i, val in enumerate(sorted(unique_rows))}
    col_scale = {val: i for i, val in enumerate(sorted(unique_cols))}

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    ax.set_xticks(range(len(unique_cols)+1))
    ax.set_yticks(range(len(unique_rows)+1))
    ax.grid(True, color='gray', linewidth=0.5)
    ax.invert_yaxis()

    # Helper to draw cells with scaled coordinates
    def draw_scaled(df, color):
        for _, row in df.iterrows():
            dp_lists = row['dp_indices']
            rg_lists = row['rg_indices']
            if all(isinstance(x, int) for x in dp_lists):
                dp_lists = [dp_lists]
                rg_lists = [rg_lists]
            for dp_sub, rg_sub in zip(dp_lists, rg_lists):
                for r, c in zip(dp_sub, rg_sub):
                    r_s = row_scale[r]
                    c_s = col_scale[c]
                    rect = plt.Rectangle((c_s, r_s), 1, 1, facecolor=color, edgecolor='gray')
                    ax.add_patch(rect)

    # Draw Matrix1-only (red)
    draw_scaled(matrix1_only, 'red')
    # Draw Matrix2-only (blue)
    draw_scaled(matrix2_only, 'blue')

    # Draw overlap
    for _, row in overlap.iterrows():
        dp_lists = row['dp_indices']
        rg_lists = row['rg_indices']
        coords1_set = set(zip(dp_lists[0], rg_lists[0]))
        coords2_set = set(zip(dp_lists[1], rg_lists[1]))
        intersection = coords1_set & coords2_set
        full_overlap = len(intersection) == len(coords1_set) == len(coords2_set)

        if full_overlap:
            draw_scaled(pd.DataFrame([row]), 'purple')
        else:
            # Partial overlap: separate
            dp1, rg1 = dp_lists[0], rg_lists[0]
            dp2, rg2 = dp_lists[1], rg_lists[1]

            dp1_overlap, rg1_overlap, dp1_non, rg1_non = [], [], [], []
            for r, c in zip(dp1, rg1):
                if (r, c) in intersection:
                    dp1_overlap.append(r)
                    rg1_overlap.append(c)
                else:
                    dp1_non.append(r)
                    rg1_non.append(c)
            dp2_overlap, rg2_overlap, dp2_non, rg2_non = [], [], [], []
            for r, c in zip(dp2, rg2):
                if (r, c) in intersection:
                    dp2_overlap.append(r)
                    rg2_overlap.append(c)
                else:
                    dp2_non.append(r)
                    rg2_non.append(c)
            # Draw
            draw_scaled(pd.DataFrame([{'dp_indices': dp1_non, 'rg_indices': rg1_non}]), 'red')
            draw_scaled(pd.DataFrame([{'dp_indices': dp2_non, 'rg_indices': rg2_non}]), 'blue')
            draw_scaled(pd.DataFrame([{'dp_indices': dp1_overlap, 'rg_indices': rg1_overlap}]), 'purple')

    ax.set_title("Targets Comparison (Dynamic Grid)")
    ax.set_xlabel("Range Index (scaled)")
    ax.set_ylabel("Doppler Index (scaled)")
    plt.show()
