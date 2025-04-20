'''
dataset.py: This script is used to load the preprocessed data from the torch.FloatTensor files and return a dataset.
'''

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

from ..features.build_features import FeatureBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FallDetectionDataset(Dataset):
    def __init__(
        self,
        processed_data_dir: str,
        split: Optional[str] = None,
        subjects: Optional[list] = None
    ):
        """Initialize the dataset by finding preprocessed sequence files.
        
        Args:
            processed_data_dir: Directory containing the processed sequence files (*.pt)
            split: Optional split name (e.g., 'train', 'val', 'test')
            subjects: Optional list of subject IDs to include
        """
        self.processed_data_dir = Path(processed_data_dir)
        
        # Find all preprocessed sequence files
        self.sequence_files = sorted(list(self.processed_data_dir.glob("*.pt")))
        
        if not self.sequence_files:
            raise FileNotFoundError(
                f"No processed sequence files found in {processed_data_dir}"
            )
        
        # Filter by split if specified
        if split:
            self.sequence_files = [
                f for f in self.sequence_files
                if split in f.stem
            ]
        
        # Filter by subjects if specified
        if subjects:
            self.sequence_files = [
                f for f in self.sequence_files
                if any(subject in f.stem for subject in subjects)
            ]
        
        logger.info(f"Loaded {len(self.sequence_files)} sequences")
    
    def __len__(self) -> int:
        """Get the number of sequences."""
        return len(self.sequence_files)
    
    def __getitem__(self, idx: int) -> Dict:
        """Load and return a preprocessed sequence.
        
        Args:
            idx: Index of the sequence file.
            
        Returns:
            Dictionary containing 'features' and 'label' tensors.
        """
        try:
            # Load the pre-saved tensor dictionary from the file
            sequence_data = torch.load(self.sequence_files[idx])
            return {
                'features': sequence_data['features'],  # Already a FloatTensor
                'label': sequence_data['label'],        # Already a FloatTensor
                'metadata': sequence_data['metadata']   # Additional metadata
            }
        except Exception as e:
            logger.error(
                f"Error loading sequence {self.sequence_files[idx]}: {str(e)}"
            )
            raise 