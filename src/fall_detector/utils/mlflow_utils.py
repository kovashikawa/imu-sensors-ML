import mlflow
import mlflow.pytorch
import torch
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class MLflowTracker:
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the MLflow tracker with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        
        # Set experiment name
        self.experiment_name = self.config['mlflow']['experiment_name']
        mlflow.set_experiment(self.experiment_name)
    
    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """Start a new MLflow run.
        
        Args:
            run_name: Optional name for the run
            
        Returns:
            Active MLflow run
        """
        return mlflow.start_run(run_name=run_name)
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameters to log
        """
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for the metrics
        """
        mlflow.log_metrics(metrics, step=step)
    
    def log_model(self, model: torch.nn.Module, model_name: str):
        """Log a PyTorch model to MLflow.
        
        Args:
            model: PyTorch model to log
            model_name: Name for the logged model
        """
        mlflow.pytorch.log_model(model, model_name)
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """Log artifacts to MLflow.
        
        Args:
            local_dir: Local directory containing artifacts
            artifact_path: Optional path within the artifact store
        """
        mlflow.log_artifacts(local_dir, artifact_path)
    
    def log_config(self, config_path: str):
        """Log configuration file to MLflow.
        
        Args:
            config_path: Path to the configuration file
        """
        mlflow.log_artifact(config_path, "config")
    
    def log_metrics_dict(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """Calculate and log classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities
            
        Returns:
            Dictionary of calculated metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_prob)
        }
        
        self.log_metrics(metrics)
        return metrics
    
    def log_model_info(self, model_info: Dict[str, Any]):
        """Log model architecture information.
        
        Args:
            model_info: Dictionary containing model information
        """
        self.log_params(model_info)
    
    def end_run(self):
        """End the current MLflow run."""
        mlflow.end_run() 