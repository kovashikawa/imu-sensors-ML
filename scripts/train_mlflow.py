'''
train_mlflow.py: This script is used to train the model and log the metrics to MLflow.
'''

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import mlflow
import mlflow.pytorch
from pathlib import Path
import logging
import argparse
from datetime import datetime
import yaml
import sys
from pathlib import Path

# Ensure that "src/" is on the import path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fall_detector.models.model import FallDetectionLSTM
from fall_detector.utils.config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SequenceDataset(Dataset):
    """Simple dataset that loads all sequence_*.pt files from a directory.
    
    Args:
        processed_dir: Directory containing the processed sequence files
    """
    def __init__(self, processed_dir: str):
        self.paths = list(Path(processed_dir).rglob("sequence_*.pt"))
        if not self.paths:
            raise FileNotFoundError(f"No .pt files found in {processed_dir}")
        logger.info(f"Found {len(self.paths)} sequence files")
        
        # Check for NaN/Inf in the first few sequences
        for i in range(min(5, len(self.paths))):
            sample = torch.load(self.paths[i])
            if torch.isnan(sample['features']).any() or torch.isinf(sample['features']).any():
                logger.warning(f"NaN/Inf found in {self.paths[i]}")
                sample['features'] = sample['features'].nan_to_num(0.)
            if torch.isnan(sample['label']).any() or torch.isinf(sample['label']).any():
                logger.warning(f"NaN/Inf found in labels of {self.paths[i]}")
                sample['label'] = sample['label'].nan_to_num(0.)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Each .pt file is a dict with 'features', 'label', 'metadata'
        sample = torch.load(self.paths[idx])
        # Ensure features and labels are clean
        sample['features'] = sample['features'].nan_to_num(0.)
        sample['label'] = sample['label'].nan_to_num(0.)
        return sample

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    experiment_name: str
) -> None:
    """Train the model and log metrics to MLflow."""
    # Set up MLflow
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_params({
            "num_epochs": num_epochs,
            "batch_size": train_loader.batch_size,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "optimizer": optimizer.__class__.__name__,
            "loss_function": criterion.__class__.__name__,
            "device": str(device)
        })
        
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                features = batch['features'].to(device)  # (batch, seq, feat)
                labels = batch['label'].float().to(device).squeeze(-1)  # (batch,)
                
                # Ensure labels are 0.0 or 1.0
                labels = labels.clamp(0, 1)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                logits = model(features).squeeze(-1)  # (batch,)
                loss = criterion(logits, labels)
                
                # Backward pass and optimize
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                probs = torch.sigmoid(logits)  # Convert to probabilities for metrics
                predicted = (probs > 0.5).float()
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Calculate training metrics
            train_loss /= len(train_loader)
            train_accuracy = 100 * train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(device)
                    labels = batch['label'].float().to(device).squeeze(-1)
                    labels = labels.clamp(0, 1)
                    
                    logits = model(features).squeeze(-1)
                    loss = criterion(logits, labels)
                    
                    val_loss += loss.item()
                    probs = torch.sigmoid(logits)
                    predicted = (probs > 0.5).float()
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate validation metrics
            val_loss /= len(val_loader)
            val_accuracy = 100 * val_correct / val_total
            
            # Log metrics
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            }, step=epoch)
            
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
            )
        
        # Save the trained model
        mlflow.pytorch.log_model(model, "model")

def main():
    """Main function to run the training pipeline."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train a fall detection model with MLflow tracking")
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
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
        
        # Set device - try MPS (Apple Silicon) first, then CUDA, then CPU
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple MPS (Metal Performance Shaders)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using CUDA")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        
        # Create datasets
        logger.info("Loading datasets...")
        processed_dir = Path(config['data']['processed_dir'])
        
        # Create dataset from all processed sequences
        dataset = SequenceDataset(processed_dir)
        
        # Split dataset into train and validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        logger.info(f"Dataset split: {len(train_dataset)} training, {len(val_dataset)} validation sequences")
        
        # Create data loaders with increased batch size if using MPS/CUDA
        batch_size = config['training']['batch_size']
        if device.type != "cpu":
            batch_size = min(batch_size * 2, 64)  # Double batch size but cap at 64
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Get feature dimension from the first sequence
        first_sequence = dataset[0]
        input_size = first_sequence['features'].shape[1]
        
        # Initialize model
        logger.info("Initializing model...")
        model = FallDetectionLSTM(
            input_size=input_size,
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout']
        ).to(device)
        
        # Initialize loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate']
        )
        
        # Train model
        logger.info("Starting training...")
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=config['training']['num_epochs'],
            device=device,
            experiment_name=config['mlflow']['experiment_name']
        )
        
        logger.info("Training completed!")
        
    except Exception as e:
        logger.exception(f"Error during training")
        raise

if __name__ == "__main__":
    main() 