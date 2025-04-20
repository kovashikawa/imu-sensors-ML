import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np

from src.fall_detector.training.train import Trainer
from src.fall_detector.data_handling.dataset import FallDetectionDataset
from src.fall_detector.features.build_features import FeatureBuilder

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize feature builder
    feature_builder = FeatureBuilder()
    
    # Load and preprocess data
    dataset = FallDetectionDataset(
        data_dir="data/01_raw",
        feature_builder=feature_builder,
        sequence_length=100,
        step=50
    )
    
    # Split dataset into train, validation, and test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4
    )
    
    # Initialize trainer
    trainer = Trainer()
    
    # Train the model
    trainer.train(train_loader, val_loader)
    
    # Save the final model
    os.makedirs("models", exist_ok=True)
    trainer.save_model("models/final_model.pt")

if __name__ == "__main__":
    main() 