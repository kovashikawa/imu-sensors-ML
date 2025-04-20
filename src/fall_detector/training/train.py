import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any
import yaml
from pathlib import Path
import time
from datetime import datetime

from ..models.model import FallDetectionLSTM
from ..utils.mlflow_utils import MLflowTracker
from ..data_handling.dataset import FallDetectionDataset

class Trainer:
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the trainer with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize MLflow tracker
        self.mlflow_tracker = MLflowTracker(config_path)
        
        # Training parameters
        self.batch_size = self.config['training']['batch_size']
        self.num_epochs = self.config['training']['num_epochs']
        self.learning_rate = self.config['training']['learning_rate']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = FallDetectionLSTM(config_path).to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.BCELoss()
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train the model for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            predicted = (output > 0.5).float()
            correct += (predicted == target).sum().item()
            total += target.size(0)
        
        return {
            'train_loss': total_loss / len(train_loader),
            'train_accuracy': correct / total
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                predicted = (output > 0.5).float()
                correct += (predicted == target).sum().item()
                total += target.size(0)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(output.cpu().numpy())
        
        metrics = {
            'val_loss': total_loss / len(val_loader),
            'val_accuracy': correct / total
        }
        
        # Calculate additional metrics using MLflow tracker
        additional_metrics = self.mlflow_tracker.log_metrics_dict(
            np.array(all_targets),
            np.array(all_preds),
            np.array(all_probs)
        )
        
        metrics.update(additional_metrics)
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
        """
        # Start MLflow run
        run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with self.mlflow_tracker.start_run(run_name=run_name):
            # Log configuration and model info
            self.mlflow_tracker.log_config("configs/config.yaml")
            self.mlflow_tracker.log_model_info(self.model.get_model_info())
            
            # Log training parameters
            self.mlflow_tracker.log_params({
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs,
                'learning_rate': self.learning_rate,
                'device': str(self.device)
            })
            
            best_val_loss = float('inf')
            for epoch in range(self.num_epochs):
                start_time = time.time()
                
                # Train epoch
                train_metrics = self.train_epoch(train_loader)
                
                # Validate
                val_metrics = self.validate(val_loader)
                
                # Log metrics
                self.mlflow_tracker.log_metrics(train_metrics, step=epoch)
                self.mlflow_tracker.log_metrics(val_metrics, step=epoch)
                
                # Save best model
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    self.mlflow_tracker.log_model(self.model, "best_model")
                
                # Print progress
                epoch_time = time.time() - start_time
                print(f"Epoch {epoch+1}/{self.num_epochs} - "
                      f"Train Loss: {train_metrics['train_loss']:.4f} - "
                      f"Val Loss: {val_metrics['val_loss']:.4f} - "
                      f"Time: {epoch_time:.2f}s")
            
            # Log final model
            self.mlflow_tracker.log_model(self.model, "final_model")
    
    def save_model(self, path: str):
        """Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
    
    def load_model(self, path: str):
        """Load a model from disk.
        
        Args:
            path: Path to the saved model
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 