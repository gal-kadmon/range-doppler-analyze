# main.py
# -----------------------------------------------------------------------------
# Entry point: generate synthetic Range-Doppler data, compare matrices, visualize results
# -----------------------------------------------------------------------------

from generate_data import generate_synthetic_rd_data
from compare_matrices import compare_target_matrices
from visualize_comparison import plot_targets


def main():
    # --------------------------
    # Generate synthetic data
    # --------------------------
    print("Generating synthetic Range-Doppler data...")
    pdu = generate_synthetic_rd_data(
        matrix_shape=(32, 1024),
        num_unique_m1=6,
        num_unique_m2=6,
        num_shared=4,
        seed=42
    )
    print("✅ Synthetic data generated. Log written to 'synthetic_generation_log.txt'.\n")

    # --------------------------
    # Compare matrices
    # --------------------------
    print("Comparing matrices...")
    pdu = compare_target_matrices(pdu, connectivity=1)
    print("✅ Comparison complete.\n")

    # --------------------------
    # Display results
    # --------------------------
    for table in pdu.tables:
        print(f"\n=== Table: {table.name} ===")
        if table.data.empty:
            print("(empty)")
        else:
            print(table.data.to_string(index=False))

    # --------------------------
    # Visualize comparison results
    # --------------------------
    matrix1_only = next(t.data for t in pdu.tables if t.name == 'matrix1_only')
    matrix2_only = next(t.data for t in pdu.tables if t.name == 'matrix2_only')
    overlap = next(t.data for t in pdu.tables if t.name == 'overlap')
    shape = pdu.tensors[0].data.shape

    print("\nDisplaying comparison visualization...")
    plot_targets(matrix1_only, matrix2_only, overlap, shape)
    print("✅ Visualization complete.")


if __name__ == "__main__":
    main()
