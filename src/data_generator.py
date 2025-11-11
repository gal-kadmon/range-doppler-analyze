import numpy as np
import pandas as pd
from typing import List, Tuple
import random
import os
import uuid

from procsess_data_unit import Process_Data_Unit, ndarray_data, base_data_attr

def generate_random_target_coords(matrix_shape: Tuple[int, int], size: int) -> List[Tuple[int, int]]:
    """Generate a random connected shape (4-connectivity) of given size."""
    coords = set()
    # Start with a random seed point
    start = (np.random.randint(0, matrix_shape[0]), np.random.randint(0, matrix_shape[1]))
    coords.add(start)
    
    while len(coords) < size:
        current = random.choice(list(coords))
        neighbors = [
            (current[0] + 1, current[1]),
            (current[0] - 1, current[1]),
            (current[0], current[1] + 1),
            (current[0], current[1] - 1)
        ]
        # Keep neighbors within bounds
        neighbors = [(r, c) for r, c in neighbors if 0 <= r < matrix_shape[0] and 0 <= c < matrix_shape[1]]
        if neighbors:
            coords.add(random.choice(neighbors))
    return list(coords)

def generate_synthetic_rd_data(
    matrix_shape=(32, 1024),
    num_targets_1=10,
    num_targets_2=10,
    overlap_targets=4,
    seed: int = None,
) -> Process_Data_Unit:
    """Generate two synthetic Range-Doppler matrices with random targets, including overlaps."""
    
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Initialize empty matrices
    matrix1 = np.zeros(matrix_shape, dtype=int)
    matrix2 = np.zeros(matrix_shape, dtype=int)
    
    log_lines = []
    
    targets_matrix1 = []
    targets_matrix2 = []
    shared_targets = []

    # Generate unique targets for matrix1
    for t in range(num_targets_1 - overlap_targets):
        size = np.random.randint(1, 13)
        coords = generate_random_target_coords(matrix_shape, size)
        for r, c in coords:
            matrix1[r, c] = 1
        targets_matrix1.append(coords)
        log_lines.append(f"Target {t+1} [Matrix 1]: {coords}")
    
    # Generate unique targets for matrix2
    for t in range(num_targets_2 - overlap_targets):
        size = np.random.randint(1, 13)
        coords = generate_random_target_coords(matrix_shape, size)
        for r, c in coords:
            matrix2[r, c] = 1
        targets_matrix2.append(coords)
        log_lines.append(f"Target {t+1} [Matrix 2]: {coords}")
    
    # Generate overlapping targets
    for t in range(overlap_targets):
        size = np.random.randint(1, 13)
        coords1 = generate_random_target_coords(matrix_shape, size)
        
        # Decide randomly whether overlap is full or partial
        full_overlap = random.choice([True, False])
        if full_overlap:
            coords2 = coords1.copy()
        else:
            coords2 = coords1.copy()
            # Remove some cells and optionally shift some cells
            if len(coords2) > 1:
                num_remove = random.randint(1, len(coords2)-1)
                coords2 = coords2[num_remove:]
                coords2_shifted = []
                for r, c in coords2:
                    dr = random.choice([0, 1, -1])
                    dc = random.choice([0, 1, -1])
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < matrix_shape[0] and 0 <= nc < matrix_shape[1]:
                        coords2_shifted.append((nr, nc))
                coords2 = coords2_shifted
        
        # Fill matrices
        for r, c in coords1:
            matrix1[r, c] = 1
        for r, c in coords2:
            matrix2[r, c] = 1
        
        shared_targets.append((coords1, coords2))
        log_lines.append(f"Target {t+1} [Shared]: Matrix1 {coords1} / Matrix2 {coords2} / Full overlap: {full_overlap}")
    
    # Wrap matrices as ndarray_data
    tensor1 = ndarray_data(
        name="matrix1",
        data=matrix1,
        ndims=2,
        dims=list(matrix1.shape),
        labels=["DP", "RG"],
        units="binary"
    )
    tensor2 = ndarray_data(
        name="matrix2",
        data=matrix2,
        ndims=2,
        dims=list(matrix2.shape),
        labels=["DP", "RG"],
        units="binary"
    )
    
    attributes = base_data_attr(
        name="synthetic_rd_data",
        uuid=str(uuid.uuid4()),
        metadata={"description": "Synthetic Range-Doppler data with random targets"}
    )
    
    # Write generation log
    log_path = os.path.join(os.getcwd(), "synthetic_generation_log.txt")
    with open(log_path, "w") as f:
        f.write("===== Synthetic Data Generation Log =====\n")
        for line in log_lines:
            f.write(line + "\n")
        f.write("=========================================\n")
    
    return Process_Data_Unit(
        attributes=attributes,
        tensors=[tensor1, tensor2],
        tables=None
    )
