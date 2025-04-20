'''
load_data.py: This script is used to load the data from the Excel files and organize it by trial.
'''

import os
import glob
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import yaml
from pathlib import Path
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all log levels
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, config: Union[str, Dict] = "configs/config.yaml"):
        """Initialize the DataLoader with configuration.
        
        Args:
            config: Either a path to the configuration file or a config dictionary
        """
        if isinstance(config, str):
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config
        
        self.raw_dir = self.config['data']['raw_dir']
        self.categories = self.config['data_loading']['categories']
        self.subject_pattern = self.config['data_loading']['subject_pattern']
        self.trial_pattern = self.config['data_loading']['trial_pattern']
        
        # Define column name mapping for normalization
        self.column_mapping = {
            # Accelerometer variants
            'accx': 'acc_x',
            'accelx': 'acc_x',
            'accelerometerx': 'acc_x',
            'acc_x_g': 'acc_x',
            'accelerometer_x': 'acc_x',
            'acceleration_x': 'acc_x',
            'x_acc': 'acc_x',
            
            'accy': 'acc_y',
            'accely': 'acc_y',
            'accelerometery': 'acc_y',
            'acc_y_g': 'acc_y',
            'accelerometer_y': 'acc_y',
            'acceleration_y': 'acc_y',
            'y_acc': 'acc_y',
            
            'accz': 'acc_z',
            'accelz': 'acc_z',
            'accelerometerz': 'acc_z',
            'acc_z_g': 'acc_z',
            'accelerometer_z': 'acc_z',
            'acceleration_z': 'acc_z',
            'z_acc': 'acc_z',
            
            # Gyroscope variants
            'gyrx': 'gyr_x',
            'gyroscopex': 'gyr_x',
            'gyro_x': 'gyr_x',
            'gyr_x_degs': 'gyr_x',
            'gyroscope_x': 'gyr_x',
            'x_gyr': 'gyr_x',
            
            'gyry': 'gyr_y',
            'gyroscopey': 'gyr_y',
            'gyro_y': 'gyr_y',
            'gyr_y_degs': 'gyr_y',
            'gyroscope_y': 'gyr_y',
            'y_gyr': 'gyr_y',
            
            'gyrz': 'gyr_z',
            'gyroscopez': 'gyr_z',
            'gyro_z': 'gyr_z',
            'gyr_z_degs': 'gyr_z',
            'gyroscope_z': 'gyr_z',
            'z_gyr': 'gyr_z'
        }
    
    def _get_file_paths(self) -> Dict[str, List[str]]:
        """Get all Excel file paths organized by category.
        
        Returns:
            Dictionary with categories as keys and lists of file paths as values
        """
        file_paths = {category: [] for category in self.categories}
        
        # Find all subject directories
        subject_dirs = glob.glob(os.path.join(self.raw_dir, self.subject_pattern))
        
        for subject_dir in subject_dirs:
            for category in self.categories:
                category_dir = os.path.join(subject_dir, category)
                if os.path.exists(category_dir):
                    # Find all trial files in the category directory
                    trial_files = glob.glob(os.path.join(category_dir, self.trial_pattern))
                    file_paths[category].extend(trial_files)
        
        return file_paths
    
    def _load_excel_file(self, file_path: str) -> pd.DataFrame:
        """Load a single Excel file and add metadata columns.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            DataFrame containing the sensor data with metadata
        """
        # Extract metadata from file path
        path_parts = Path(file_path).parts
        subject = path_parts[-3]  # e.g., 'sub1'
        category = path_parts[-2]  # e.g., 'ADLs'
        trial = Path(file_path).stem  # e.g., 'AXR_AS_trial1'
        
        # Load the Excel file (read first sheet by default)
        df = pd.read_excel(file_path)
        
        # Normalize column names:
        # 1. Basic normalization: strip, lowercase, replace non-word chars with underscores
        cols = (
            df.columns
              .str.strip()
              .str.lower()
              .str.replace(r'[^\w]', '_', regex=True)
              .str.replace(r'__+', '_', regex=True)  # Collapse multiple underscores
              .str.strip('_')                        # Remove leading/trailing underscores
        )
        
        # 2. Apply the mapping for known variants
        normalized_cols = pd.Series(cols).replace(self.column_mapping).tolist()
        
        # 3. Handle body-site prefixes and units
        # Convert "r_ankle_acceleration_x_m_s_2" -> "acc_x"
        # Convert "l_thigh_angular_velocity_z_rad_s" -> "gyr_z"
        normalized_cols = [
            re.sub(r'.+?_acceleration_([xyz])_m_s_2$', r'acc_\1', col)
            for col in normalized_cols
        ]
        normalized_cols = [
            re.sub(r'.+?_angular_velocity_([xyz])_rad_s$', r'gyr_\1', col)
            for col in normalized_cols
        ]
        
        # 4. Log the column name mapping for debugging
        for old, new in zip(df.columns, normalized_cols):
            if old != new:
                logger.debug(f"Renamed column: '{old}' -> '{new}'")
        
        # 5. Apply the new column names
        df.columns = normalized_cols
        
        # 6. Drop any duplicate columns (keeps first occurrence)
        if df.columns.duplicated().any():
            dupes = df.columns[df.columns.duplicated()].unique()
            logger.warning(f"Duplicate columns {list(dupes)} detected in {file_path}; dropping duplicates.")
            df = df.loc[:, ~df.columns.duplicated()]
        
        logger.debug(f"Columns after deduplication for {file_path}: {df.columns.tolist()}")
        
        # Add metadata columns
        df['subject'] = subject
        df['category'] = category
        df['trial'] = trial
        df['fall_label'] = self.config['target']['label_mapping'][category]
        
        return df
    
    def load_all_data(self, max_trials: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Load all data from Excel files and organize by trial.
        
        Args:
            max_trials: If set, only load the first N trials
            
        Returns:
            Dictionary mapping trial names to their data
        """
        file_paths = self._get_file_paths()
        
        # Flatten into a single list of (category, path)
        all_files = []
        for cat, paths in file_paths.items():
            for p in paths:
                all_files.append((cat, p))
        
        if max_trials is not None:
            all_files = all_files[:max_trials]
            logger.info(f"Limiting load to first {max_trials} files")
        
        trial_data = {}
        skipped_files = []
        
        for category, path in all_files:
            try:
                # Load the Excel file
                df = self._load_excel_file(path)
                
                # Get trial name from the file path
                trial_name = Path(path).stem
                
                # Store the data
                trial_data[trial_name] = df
                
                logger.info(f"Successfully loaded {trial_name}")
                
            except Exception as e:
                logger.error(f"Error loading {path}: {str(e)}")
                skipped_files.append((path, str(e)))
                continue
        
        # Log skipped files
        if skipped_files:
            logger.warning(f"Skipped {len(skipped_files)} files due to errors")
            for path, error in skipped_files:
                logger.debug(f"Skipped {path}: {error}")
        
        return trial_data
    
    def get_data_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate basic statistics about the loaded data.
        
        Args:
            df: DataFrame containing the loaded data
            
        Returns:
            Dictionary with basic statistics
        """
        stats = {
            'total_samples': len(df),
            'subjects': df['subject'].nunique(),
            'categories': df['category'].value_counts().to_dict(),
            'trials': df['trial'].nunique(),
            'fall_labels': df['fall_label'].value_counts().to_dict()
        }
        
        return stats
