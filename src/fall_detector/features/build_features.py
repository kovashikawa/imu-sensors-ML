import numpy as np
import pandas as pd
from typing import List, Dict, Union
import yaml
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureBuilder:
    def __init__(self, config: Union[str, Dict] = "configs/config.yaml"):
        """Initialize the FeatureBuilder with configuration.
        
        Args:
            config: Either a path to the configuration file or a config dictionary
        """
        if isinstance(config, str):
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config
        
        self.selected_features = self.config['features']['selected_features']
    
    def calculate_magnitude(self, df: pd.DataFrame, axis: str) -> pd.Series:
        """Calculate the magnitude of a 3D vector.
        
        Args:
            df: DataFrame containing the sensor data
            axis: Base name of the axis (e.g., 'acc' or 'gyr')
            
        Returns:
            Series containing the magnitude values
        """
        # Check if required columns exist
        axes = [f'{axis}_x', f'{axis}_y', f'{axis}_z']
        missing = [ax for ax in axes if ax not in df.columns]
        if missing:
            raise KeyError(f"Missing columns for magnitude: {missing!r}")
        
        # Use vectorized operations for better performance
        return np.sqrt((df[axes] ** 2).sum(axis=1))
    
    def calculate_rolling_stats(self, series: pd.Series, window: int = 5) -> pd.DataFrame:
        """Calculate rolling statistics for a time series.
        
        Args:
            series: Input time series
            window: Window size for rolling calculations
            
        Returns:
            DataFrame with rolling statistics
        """
        return pd.DataFrame({
            f'{series.name}_mean_{window}': series.rolling(window=window).mean(),
            f'{series.name}_std_{window}': series.rolling(window=window).std()
        })
    
    def calculate_motion_intensity(self, df: pd.DataFrame) -> pd.Series:
        """Calculate motion intensity from accelerometer and gyroscope data.
        
        Args:
            df: DataFrame containing the sensor data
            
        Returns:
            Series containing the motion intensity values
        """
        # Check if required columns exist
        required_columns = ['acc_mag', 'gyr_mag']
        for col in required_columns:
            if col not in df.columns:
                raise KeyError(f"Required column '{col}' not found in DataFrame. Available columns: {df.columns.tolist()}")
        
        acc_mag = df['acc_mag']
        gyr_mag = df['gyr_mag']
        
        # Normalize the magnitudes
        acc_mag_norm = (acc_mag - acc_mag.mean()) / acc_mag.std()
        gyr_mag_norm = (gyr_mag - gyr_mag.mean()) / gyr_mag.std()
        
        # Combine normalized magnitudes
        return np.sqrt(acc_mag_norm**2 + gyr_mag_norm**2)
    
    def calculate_orientation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate roll and pitch angles from accelerometer data.
        
        Args:
            df: DataFrame containing the sensor data
            
        Returns:
            DataFrame with roll and pitch angles
        """
        # Check if required columns exist
        required_columns = ['acc_x', 'acc_y', 'acc_z']
        for col in required_columns:
            if col not in df.columns:
                raise KeyError(f"Required column '{col}' not found in DataFrame. Available columns: {df.columns.tolist()}")
        
        acc_x = df['acc_x']
        acc_y = df['acc_y']
        acc_z = df['acc_z']
        
        # Calculate roll and pitch
        roll = np.arctan2(acc_y, np.sqrt(acc_x**2 + acc_z**2))
        pitch = np.arctan2(-acc_x, np.sqrt(acc_y**2 + acc_z**2))
        
        return pd.DataFrame({
            'roll': roll,
            'pitch': pitch
        })
    
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build all features for the dataset.
        
        Args:
            df: DataFrame containing the raw sensor data
            
        Returns:
            DataFrame with all engineered features
        """
        # Log columns before processing
        logger.debug(f"Columns in raw trial data: {df.columns.tolist()}")
        
        # Calculate magnitudes
        df['acc_mag'] = self.calculate_magnitude(df, 'acc')
        df['gyr_mag'] = self.calculate_magnitude(df, 'gyr')
        
        # Calculate rolling statistics
        acc_stats = self.calculate_rolling_stats(df['acc_mag'])
        gyr_stats = self.calculate_rolling_stats(df['gyr_mag'])
        df = pd.concat([df, acc_stats, gyr_stats], axis=1)
        
        # Calculate motion intensity
        df['motion_intensity'] = self.calculate_motion_intensity(df)
        
        # Calculate orientation
        orientation = self.calculate_orientation(df)
        df = pd.concat([df, orientation], axis=1)
        
        # Check if all required features are present
        for feature in self.selected_features:
            if feature not in df.columns:
                raise KeyError(f"Expected feature column '{feature}' not found in DataFrame. Available columns: {df.columns.tolist()}")
        
        # Select only the required features
        return df[self.selected_features + ['fall_label']]
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get descriptions for all engineered features.
        
        Returns:
            Dictionary mapping feature names to their descriptions
        """
        return {
            'acc_x': 'Accelerometer X-axis',
            'acc_y': 'Accelerometer Y-axis',
            'acc_z': 'Accelerometer Z-axis',
            'gyr_x': 'Gyroscope X-axis',
            'gyr_y': 'Gyroscope Y-axis',
            'gyr_z': 'Gyroscope Z-axis',
            'acc_mag': 'Accelerometer magnitude',
            'gyr_mag': 'Gyroscope magnitude',
            'acc_mag_mean_5': '5-sample rolling mean of accelerometer magnitude',
            'acc_mag_std_5': '5-sample rolling standard deviation of accelerometer magnitude',
            'gyr_mag_mean_5': '5-sample rolling mean of gyroscope magnitude',
            'gyr_mag_std_5': '5-sample rolling standard deviation of gyroscope magnitude',
            'motion_intensity': 'Combined motion intensity from accelerometer and gyroscope',
            'roll': 'Roll angle from accelerometer',
            'pitch': 'Pitch angle from accelerometer'
        }
