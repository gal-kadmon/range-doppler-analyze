"""
TXT file datareader implementation.
Handles loading synthetic process units based on a list of PDU names stored in a .txt file.
Each PDU name is mapped to a deterministic synthetic dataset using a seed derived from its ID.
"""

import os
from typing import List, Dict, Any

from datareader_interface import DataReader
from process_data_unit import Process_Data_Unit
from generate_data import generate_synthetic_rd_data
from compare_matrices import compare_target_matrices


class TxtReader(DataReader):
    """Datareader for TXT files containing a list of PDU identifiers."""

    # ----------------------------------------------------------------------
    def can_handle(self, source_path: str, source_type: str) -> bool:
        """
        Returns True if source is a .txt file.
        """
        return source_type == "file" and source_path.endswith(".txt")

    # ----------------------------------------------------------------------
    def get_process_unit_names(self, source_path: str, source_type: str) -> List[str]:
        """
        Reads the list of PDU identifiers from the given TXT file.
        Each non-empty line is treated as a PDU name.
        """
        names = []

        if not os.path.exists(source_path):
            return names

        try:
            with open(source_path, "r", encoding="utf-8") as file:
                for line in file:
                    name = line.strip()
                    if name:
                        names.append(name)
        except Exception:
            pass

        return names

    # ----------------------------------------------------------------------
    def get_process_unit_metadata(self, source_path: str, source_type: str) -> List[Dict[str, Any]]:
        """
        Return minimal metadata describing the available PDUs in the TXT file.
        This does NOT load the PDUs themselves.
        """
        metadata_list = []
        names = self.get_process_unit_names(source_path, source_type)

        for name in names:
            metadata_list.append({
                "name": name,
                "uuid": "",
                "metadata": {
                    "synthetic": True,
                    "source_type": "txt_list"
                },
                "tensor_count": 0,     # Synthetic PDUs are created only upon explicit loading
                "table_count": 0,
                "tensor_names": [],
                "table_names": []
            })

        return metadata_list

    # ----------------------------------------------------------------------
    def get_process_unit_by_name(self, source_path: str, source_type: str, unit_name: str) -> Dict[str, Any]:
        """
        Generate a synthetic PDU for the requested unit_name.

        How it works:
            • Parse numeric ID from the name (e.g., "PDU_004" → seed=4)
            • Use seed to generate reproducible synthetic Range–Doppler data
            • Run compare_target_matrices to attach tables, metadata, and mask
            • Add reader metadata
            • Return PDU as a JSON-ready dict
        """
        # Validate name exists in TXT list
        available = self.get_process_unit_names(source_path, source_type)
        if unit_name not in available:
            raise FileNotFoundError(f"PDU '{unit_name}' not listed in TXT file.")

        # Extract seed (supports formats like "PDU_001", "PDU_12", etc.)
        try:
            numeric_part = ''.join([ch for ch in unit_name if ch.isdigit()])
            seed = int(numeric_part) if numeric_part else 0
        except Exception:
            seed = 0

        # Generate synthetic PDU
        pdu = generate_synthetic_rd_data(seed=seed)

        # Run your comparison algorithm
        pdu = compare_target_matrices(pdu)

        # Add metadata describing the synthetic origin
        if pdu.attributes.metadata is None:
            pdu.attributes.metadata = {}

        pdu.attributes.metadata.update({
            "synthetic": True,
            "generated_from_pdu_id": unit_name,
            "synthetic_seed": seed,
            "datareader_type": "TxtReader",
            "txt_source_path": source_path
        })

        # Return JSON-like representation
        return pdu.to_dict()
