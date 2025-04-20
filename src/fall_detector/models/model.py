'''
model.py: This script is used to define the model for the fall detection.
'''

import torch
import torch.nn as nn

class FallDetectionLSTM(nn.Module):
    """LSTM-based model for fall detection.
    
    Args:
        input_size: Number of input features
        hidden_size: Number of hidden units in LSTM
        num_layers: Number of LSTM layers
        dropout: Dropout probability
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc_logits = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            
        Returns:
            Raw logits of shape (batch,)
        """
        # x: (batch, seq_len, input_size)
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        _, (hn, _) = self.lstm(x, (h0, c0))
        last_hidden = hn[-1]                # (batch, hidden_size)
        logits = self.fc_logits(last_hidden)       # (batch, 1)
        return logits.squeeze(-1)           # (batch,)
