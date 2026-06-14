import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    """
    Computes an attention-weighted sum over the sequence dimension.
    """
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        scores = self.attention(x)  # Shape: (batch_size, seq_len, 1)
        weights = torch.softmax(scores, dim=1)  # Shape: (batch_size, seq_len, 1)
        pooled = torch.sum(weights * x, dim=1)  # Shape: (batch_size, input_dim)
        return pooled, weights

class BaselineMLP(nn.Module):
    """
    Baseline classifier: Mean Pooling + MLP.
    """
    def __init__(self, input_dim=2304, num_classes=2, dropout=0.3):
        super(BaselineMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1024),  # Wait, let's look at requirements:
            # Linear(1024) -> ReLU -> Dropout -> Linear(256) -> ReLU -> Linear(2)
        )
        # Correcting design to match prompt:
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # Apply mean pooling over the temporal (sequence) dimension
        mean_pooled = torch.mean(x, dim=1)  # Shape: (batch_size, input_dim)
        logits = self.classifier(mean_pooled)
        return logits

class AttentionMLP(nn.Module):
    """
    Improved Classifier Option A: Attention Pooling + MLP.
    """
    def __init__(self, input_dim=2304, num_classes=2, dropout=0.3):
        super(AttentionMLP, self).__init__()
        self.pool = AttentionPooling(input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        pooled, weights = self.pool(x)  # Shape: (batch_size, input_dim)
        logits = self.classifier(pooled)
        return logits

class BiLSTMClassifier(nn.Module):
    """
    Improved Classifier Option B: BiLSTM + Attention Pooling + MLP.
    """
    def __init__(self, input_dim=2304, lstm_hidden=128, num_classes=2, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        # BiLSTM outputs double the hidden size due to bidirectionality
        lstm_out_dim = lstm_hidden * 2
        
        self.pool = AttentionPooling(lstm_out_dim)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # Shape: (batch_size, seq_len, lstm_out_dim)
        pooled, weights = self.pool(lstm_out)  # Shape: (batch_size, lstm_out_dim)
        logits = self.classifier(pooled)
        return logits

if __name__ == '__main__':
    # Simple shape verification
    batch_size = 4
    seq_len = 100
    input_dim = 2304
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    m1 = BaselineMLP()
    m2 = AttentionMLP()
    m3 = BiLSTMClassifier()
    
    print("Baseline MLP output shape:", m1(x).shape)
    print("Attention MLP output shape:", m2(x).shape)
    print("BiLSTM Classifier output shape:", m3(x).shape)
    print("All models verified successfully!")
