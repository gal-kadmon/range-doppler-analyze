from data_generator import generate_synthetic_rd_data
from compare_matrices import compare_target_matrices
import pandas as pd

def main():
    # --------------------------
    # Generate synthetic data
    # --------------------------
    print("Generating synthetic Range-Doppler data...")
    pdu = generate_synthetic_rd_data(
        matrix_shape=(32, 1024),
        num_targets_1=10,
        num_targets_2=10,
        overlap_targets=4,
        seed=42
    )
    print("Synthetic data generated. Log written to 'synthetic_generation_log.txt'.\n")

    # --------------------------
    # Compare matrices
    # --------------------------
    print("Comparing matrices...")
    pdu = compare_target_matrices(pdu)
    print("Comparison complete. Report written to 'comparison_report.txt'.\n")

    # --------------------------
    # Display results
    # --------------------------
    for table in pdu.tables:
        print(f"=== Table: {table.name} ===")
        print(table.data.to_string(index=False))

if __name__ == "__main__":
    main()
