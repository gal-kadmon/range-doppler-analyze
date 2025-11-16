"""
Abstract base class interface for datareaders.
All datareaders must implement this interface.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from process_data_unit import Process_Data_Unit


class DataReader(ABC):
    """Abstract base class for all datareaders"""
    
    @abstractmethod
    def get_process_unit_names(self, source_path: str, source_type: str) -> List[str]:
        """
        Get a list of all process unit names from the source.
        
        Args:
            source_path: Path to the file or folder
            source_type: 'file' or 'folder'
            
        Returns:
            List of process unit names
        """
        pass
    
    @abstractmethod
    def get_process_unit_metadata(self, source_path: str, source_type: str) -> List[Dict[str, Any]]:
        """
        Get metadata for all process units from the source.
        
        Args:
            source_path: Path to the file or folder
            source_type: 'file' or 'folder'
            
        Returns:
            List of metadata dictionaries, each containing:
            - name: str
            - uuid: str
            - metadata: dict
            - tensor_count: int
            - table_count: int
            - tensor_names: List[str]
            - table_names: List[str]
        """
        pass
    
    @abstractmethod
    def get_process_unit_by_name(self, source_path: str, source_type: str, unit_name: str) -> Dict[str, Any]:
        """
        Get a specific process unit by name.
        
        Args:
            source_path: Path to the file or folder
            source_type: 'file' or 'folder'
            unit_name: Name of the process unit to retrieve
            
        Returns:
            Process_Data_Unit as a dictionary (from to_dict())
        """
        pass
    
    @abstractmethod
    def can_handle(self, source_path: str, source_type: str) -> bool:
        """
        Check if this datareader can handle the given source.
        
        Args:
            source_path: Path to the file or folder
            source_type: 'file' or 'folder'
            
        Returns:
            True if this datareader can handle the source, False otherwise
        """
        pass


def detect_datareader_type(source_path: str, source_type: str) -> Optional[str]:
    """
    Automatically detect the datareader type from the source path.
    
    Args:
        source_path: Path to the file or folder
        source_type: 'file' or 'folder'
        
    Returns:
        Datareader type string ('npy', 'tar', 'mat', 'csv') or None if unknown
    """
    import os
    import glob
    
    if source_type == 'file':
        # Check file extension
        if source_path.endswith('.tar'):
            return 'tar'
        elif source_path.endswith('.npy'):
            return 'npy'
        elif source_path.endswith('.mat'):
            return 'mat'
        elif source_path.endswith('.csv'):
            return 'csv'
    elif source_type == 'folder':
        # Check what types of files are in the folder
        if os.path.isdir(source_path):
            # Check for tar files
            tar_files = glob.glob(os.path.join(source_path, "*.tar"))
            if tar_files:
                return 'tar'
            
            # Check for numpy files
            npy_files = glob.glob(os.path.join(source_path, "*.npy"))
            if npy_files:
                return 'npy'
            
            # Check for matlab files
            mat_files = glob.glob(os.path.join(source_path, "*.mat"))
            if mat_files:
                return 'mat'
    
    return None