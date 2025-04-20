'''
preprocess_data.py: This script is used to preprocess the data from the Excel files and save it as a torch.FloatTensor.
'''

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import logging
import argparse
from typing import Dict, List, Tuple, Optional

from src.fall_detector.data_handling.load_data import DataLoader
from src.fall_detector.features.build_features import FeatureBuilder

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all log levels
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config: Dict, max_trials: Optional[int] = None):
        """Initialize the data preprocessor.
        
        Args:
            config: Configuration dictionary
            max_trials: If set, only process the first N trials
        """
        self.config = config
        self.max_trials = max_trials
        
        # Initialize components
        self.data_loader = DataLoader(config)
        self.feature_builder = FeatureBuilder(config)
        
        # Get parameters from config
        self.sequence_length = self.config['preprocessing']['sequence_length']
        self.step = self.config['preprocessing']['step']
        self.processed_dir = Path(self.config['data']['processed_dir'])
        
        # Create processed directory if it doesn't exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def preprocess_trial(self, trial_data: pd.DataFrame, trial_name: str) -> List[Dict]:
        """Preprocess a single trial's data.
        
        Args:
            trial_data: DataFrame containing the trial's data
            trial_name: Name of the trial
            
        Returns:
            List of preprocessed sequences
        """
        # Log columns before processing
        logger.debug(f"Columns in raw trial data: {trial_data.columns.tolist()}")
        
        # Build features
        try:
            features_df = self.feature_builder.build_features(trial_data)
            logger.debug(f"Successfully built features for {trial_name}. Shape: {features_df.shape}")
        except KeyError as e:
            logger.error(f"Missing column when building features for {trial_name}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error building features for {trial_name}: {str(e)}")
            raise
        
        # Create sequences
        sequences = []
        try:
            # Use the already-parsed feature list from the FeatureBuilder
            features = features_df[self.feature_builder.selected_features].values
            labels = features_df['fall_label'].values
            
            if len(features) < self.sequence_length:
                logger.warning(
                    f"Trial {trial_name} has too few samples after processing: "
                    f"{len(features)} < {self.sequence_length}. Skipping."
                )
                return []
            
            for i in range(0, len(features) - self.sequence_length + 1, self.step):
                sequence_features = features[i:i + self.sequence_length]
                sequence_labels = labels[i:i + self.sequence_length]
                
                # Use the majority label in the sequence
                label = 1 if np.mean(sequence_labels) > 0.5 else 0
                
                sequences.append({
                    'features': torch.FloatTensor(sequence_features),
                    'label': torch.FloatTensor([label]),
                    'metadata': {
                        'trial_name': trial_name,
                        'start_idx': i,
                        'end_idx': i + self.sequence_length
                    }
                })
            
            logger.info(f"Created {len(sequences)} sequences for trial {trial_name}")
        except Exception as e:
            logger.error(f"Error creating sequences for {trial_name}: {str(e)}")
            raise
        
        return sequences
    
    def save_sequences(self, sequences: List[Dict], trial_name: str) -> None:
        """Save preprocessed sequences to disk.
        
        Args:
            sequences: List of preprocessed sequences
            trial_name: Name of the trial
        """
        if not sequences:
            logger.warning(f"No sequences to save for trial {trial_name}")
            return
        
        # Create trial directory
        trial_dir = self.processed_dir / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each sequence
        for i, sequence in enumerate(sequences):
            sequence_path = trial_dir / f"sequence_{i:04d}.pt"
            torch.save(sequence, sequence_path)
        
        logger.info(f"Saved {len(sequences)} sequences for trial {trial_name}")
    
    def process_all_data(self) -> None:
        """Process all data and save preprocessed sequences."""
        # Load all data
        try:
            all_data = self.data_loader.load_all_data(max_trials=self.max_trials)
            logger.info(f"Successfully loaded {len(all_data)} trials")
        except Exception as e:
            logger.exception(f"Error loading data")
            raise
        
        success_count = 0
        error_count = 0
        skipped_trials = []
        
        # Process each trial
        for trial_name, trial_data in all_data.items():
            logger.info(f"Processing trial: {trial_name}")
            
            try:
                # Preprocess trial
                sequences = self.preprocess_trial(trial_data, trial_name)
                
                # Save sequences
                self.save_sequences(sequences, trial_name)
                
                if sequences:
                    success_count += 1
                else:
                    skipped_trials.append((trial_name, "Too few samples after processing"))
                    error_count += 1
                
            except Exception as e:
                logger.exception(f"Error processing trial {trial_name}")
                skipped_trials.append((trial_name, str(e)))
                error_count += 1
                continue
        
        # Log skipped trials
        if skipped_trials:
            skipped_log_path = self.processed_dir / "skipped_trials.txt"
            with open(skipped_log_path, 'w') as f:
                for trial, reason in skipped_trials:
                    f.write(f"{trial}: {reason}\n")
            logger.warning(f"Skipped {len(skipped_trials)} trials. See {skipped_log_path} for details.")
        
        logger.info(f"Processing summary: {success_count} trials successful, {error_count} trials failed")

def main():
    """Main function to run the preprocessing pipeline."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Preprocess IMU data for fall detection")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml", 
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug logging"
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="If set, only preprocess the first N trials and then exit"
    )
    args = parser.parse_args()
    
    # Set logging level based on arguments
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    
    try:
        # Load configuration centrally
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {args.config}")
        
        # Initialize preprocessor with the loaded config and max_trials
        preprocessor = DataPreprocessor(config, max_trials=args.max_trials)
        
        # Process all data
        preprocessor.process_all_data()
        
        logger.info("Preprocessing completed successfully!")
        
    except Exception as e:
        logger.exception(f"Error during preprocessing")
        raise

if __name__ == "__main__":
    main() 