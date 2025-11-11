from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any
import numpy as np
import pandas as pd
import os

@dataclass
class base_data_attr:
    name: str
    uuid: str
    metadata: Optional[dict[str, Any]] = None

@dataclass
class dataframe_data:
    name: str
    data: pd.DataFrame
    metadata: Optional[dict[str, Any]] = None

@dataclass
class tensor_extra_attr:
    x: int
    y: int
    z: Optional[int] = None
    metadata: Optional[dict[str, Any]] = None

@dataclass
class ndarray_data:
    name: str
    data: np.ndarray
    ndims: int
    dims: List[int]
    labels: List[str]
    units: str
    metadata: Optional[list[tensor_extra_attr]] = None


@dataclass
class Process_Data_Unit:
    """A dataclass to encapsulate numpy arrays and pandas dataframes with metadata"""
    attributes: base_data_attr
    tensors: Optional[List[ndarray_data]] = None
    tables: Optional[List[dataframe_data]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the Process_Data_Unit to a dictionary for JSON serialization"""
        result = {
            'attributes': {
                'name': self.attributes.name,
                'uuid': self.attributes.uuid,
                'metadata': self.attributes.metadata
            }
        }
        
        if self.tensors:
            result['tensors'] = []
            for tensor in self.tensors:
                try:
                    # Check if data is complex
                    if tensor.data is not None and np.iscomplexobj(tensor.data):
                        # For complex arrays, convert to a JSON-serializable format
                        # Convert to list of [real, imag] pairs for each element
                        # This is more efficient than separate real/imag arrays
                        data_list = []
                        flat_data = tensor.data.flatten()
                        for val in flat_data:
                            data_list.append([float(val.real), float(val.imag)])
                        # Reshape back to original shape structure
                        # Store as flat list with shape info - frontend can reshape
                        tensor_dict = {
                            'name': tensor.name,
                            'data': data_list,
                            'data_shape': list(tensor.data.shape),
                            'data_dtype': 'complex',
                            'ndims': tensor.ndims,
                            'dims': tensor.dims,
                            'labels': tensor.labels,
                            'units': tensor.units
                        }
                    else:
                        # For real arrays, convert to list as before
                        data_list = tensor.data.tolist() if tensor.data is not None else None
                        tensor_dict = {
                            'name': tensor.name,
                            'data': data_list,
                            'ndims': tensor.ndims,
                            'dims': tensor.dims,
                            'labels': tensor.labels,
                            'units': tensor.units
                        }
                    if tensor.metadata:
                        tensor_dict['metadata'] = [
                            {
                                'x': attr.x,
                                'y': attr.y,
                                'z': attr.z,
                                'metadata': attr.metadata
                            } for attr in tensor.metadata
                        ]
                    result['tensors'].append(tensor_dict)
                except Exception as e:
                    print(f"Error converting tensor data to list: {str(e)}")
                    # If conversion fails, return metadata without data
                    tensor_dict = {
                        'name': tensor.name,
                        'data': None,
                        'ndims': tensor.ndims,
                        'dims': tensor.dims,
                        'labels': tensor.labels,
                        'units': tensor.units,
                        'error': f"Failed to serialize data: {str(e)}"
                    }
                    result['tensors'].append(tensor_dict)
        
        # Add tables if they exist
        if self.tables:
            result['tables'] = []
            for table in self.tables:
                try:
                    # Convert DataFrame to records (list of dicts)
                    table_dict = {
                        'name': table.name,
                        'metadata': table.metadata
                    }
                    if table.data is not None:
                        table_dict['data'] = table.data.to_dict('records')
                        table_dict['columns'] = list(table.data.columns)
                        table_dict['rows'] = len(table.data)
                    else:
                        table_dict['data'] = None
                        table_dict['columns'] = []
                        table_dict['rows'] = 0
                    result['tables'].append(table_dict)
                except Exception as e:
                    print(f"Error converting table data to dict: {str(e)}")
                    # If conversion fails, return metadata without data
                    table_dict = {
                        'name': table.name,
                        'data': None,
                        'metadata': table.metadata,
                        'error': f"Failed to serialize table data: {str(e)}"
                    }
                    result['tables'].append(table_dict)
        
        return result
    
    @classmethod
    def from_numpy_file(cls, file_path: str, filename: str, uuid: str = None) -> 'Process_Data_Unit':
        """Create a Process_Data_Unit from a numpy file"""
        import uuid as uuid_module
        
        data = np.load(file_path, allow_pickle=True)
        if not isinstance(data, np.ndarray):
            raise ValueError(f"File {filename} does not contain a valid numpy array")
        
        # Generate UUID if not provided
        if uuid is None:
            uuid = str(uuid_module.uuid4())
        
        # Create dimension labels based on array shape
        labels = [f"dim_{i}" for i in range(len(data.shape))]
        
        # Create tensor data
        tensor = ndarray_data(
            name=filename,
            data=data,
            ndims=len(data.shape),
            dims=list(data.shape),
            labels=labels,
            units="unknown"
        )
        
        # Create base attributes
        attributes = base_data_attr(
            name=filename,
            uuid=uuid,
            metadata={
                'file_path': file_path,
                'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
                'data_type': 'numpy'
            }
        )
        
        return cls(
            attributes=attributes,
            tensors=[tensor]
        )