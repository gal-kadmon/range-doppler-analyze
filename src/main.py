# main.py
# -----------------------------------------------------------------------------
# Demo for system integration using TxtReader + synthetic PDU generation.
# Works entirely with dictionary-based PDUs (as required by the DataReader interface).
# -----------------------------------------------------------------------------

import os
import pandas as pd

from txt_reader import TxtReader
from visualize_comparison import plot_targets


def main():

    # ---------------------------------------------------------
    # Step 1: Create or load TXT file with PDU names
    # ---------------------------------------------------------
    txt_path = "pdu_list.txt"

    if not os.path.exists(txt_path):
        with open(txt_path, "w") as f:
            for i in range(1, 31):
                f.write(f"PDU_{i:03d}\n")
        print(f"Created example {txt_path}")

    reader = TxtReader()

    # ---------------------------------------------------------
    # Step 2: Read list of PDU names
    # ---------------------------------------------------------
    print("\nReading PDU list from file...")
    names = reader.get_process_unit_names(txt_path, "file")

    if not names:
        print("No PDUs found in TXT file.")
        return

    print(f"Found {len(names)} available PDUs:")
    for name in names:
        print(f"    {name}")

    # ---------------------------------------------------------
    # Step 3: Simulate user selection
    # ---------------------------------------------------------
    selected_pdu = names[28]      
    print(f"\nSimulated user selection → {selected_pdu}")

    # ---------------------------------------------------------
    # Step 4: Load PDU using the reader (returns dict!)
    # ---------------------------------------------------------
    print(f"Loading PDU '{selected_pdu}' (synthetic via seed)...\n")

    pdu = reader.get_process_unit_by_name(txt_path, "file", selected_pdu)

    print("PDU successfully generated and processed.\n")

    # ---------------------------------------------------------
    # Step 5: Print tables (dictionary format)
    # ---------------------------------------------------------
    print("=== Available Tables in PDU ===")

    tables = pdu.get("tables", [])
    if not tables:
        print("(No tables found)")
    else:
        for table in tables:
            print(f"\nTable: {table.get('name', '(unknown)')}")
            df = pd.DataFrame(table.get("data", []))
            if df.empty:
                print("(empty)")
            else:
                print(df.to_string(index=False))

    # ---------------------------------------------------------
    # Step 6: Prepare DataFrames for visualization
    # ---------------------------------------------------------
    # Helper to extract table by name
    def get_table(name: str):
        return next((t for t in tables if t["name"] == name), None)

    df_m1_only = pd.DataFrame(get_table("matrix1_only")["data"])
    df_m2_only = pd.DataFrame(get_table("matrix2_only")["data"])
    df_overlap = pd.DataFrame(get_table("overlap")["data"])

    # ---------------------------------------------------------
    # Step 7: Extract matrix shape (from dict tensor info)
    # ---------------------------------------------------------
    tensors = pdu.get("tensors", [])
    if not tensors:
        raise RuntimeError("No tensors found inside PDU.")

    # tensor format:
    # {"name": ..., "data": [...], "dims": [...], ...}
    tensor0 = tensors[0]
    shape = tuple(tensor0["dims"])     # (range, doppler)

    # ---------------------------------------------------------
    # Step 8: Visualization
    # ---------------------------------------------------------
    print("\nRendering visualization...")
    plot_targets(df_m1_only, df_m2_only, df_overlap, shape)

    print("\nDone. System simulation completed.\n")


if __name__ == "__main__":
    main()
