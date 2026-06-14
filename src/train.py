import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.model import BaselineMLP, AttentionMLP, BiLSTMClassifier

class EmbeddingDataset(Dataset):
    """
    Dataset class that loads pre-extracted HuBERT embeddings from disk.
    """
    def __init__(self, metadata_path, split, proj_dir):
        self.proj_dir = proj_dir
        self.metadata = pd.read_csv(metadata_path)
        self.metadata = self.metadata[self.metadata['split'] == split].reset_index(drop=True)
        
        # Label mapping: real (genuine) -> 0, fake (deepfake) -> 1
        self.label_map = {'real': 0, 'fake': 1}
        
    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_rel = row['filepath']
        label_str = row['label']
        split = row['split']
        
        # Construct path to cached embedding
        base_name = os.path.splitext(os.path.basename(file_rel))[0]
        embedding_path = os.path.join(
            self.proj_dir, 'data', 'embeddings', split, label_str, f"{base_name}.pt"
        )
        
        # Load cached embedding tensor of shape (seq_len, 2304)
        embedding = torch.load(embedding_path)
        label = self.label_map[label_str]
        
        return embedding, label

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
    epoch_loss = running_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    return epoch_loss, acc, precision, recall, f1

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            logits = model(inputs)
            loss = criterion(logits, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    val_loss = running_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    return val_loss, acc, precision, recall, f1

def main():
    parser = argparse.ArgumentParser(description="Train classification models on cached embeddings")
    parser.add_argument(
        '--model', type=str, default='baseline', choices=['baseline', 'attention', 'lstm'],
        help="Model architecture to train (baseline, attention, lstm)"
    )
    parser.add_argument('--epochs', type=int, default=30, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--patience', type=int, default=7, help="Patience for early stopping")
    args = parser.parse_args()
    
    proj_dir = 'C:/Users/Shreshtha Shrinivas/.gemini/antigravity/scratch/deepfake-audio-detection'
    metadata_path = os.path.join(proj_dir, 'data', 'metadata_subset.csv')
    
    if not os.path.exists(metadata_path):
        # Fallback to main metadata if subset isn't created (though it should be)
        metadata_path = os.path.join(proj_dir, 'data', 'metadata.csv')
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Load datasets
    train_dataset = EmbeddingDataset(metadata_path, 'training', proj_dir)
    val_dataset = EmbeddingDataset(metadata_path, 'validation', proj_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Model selection
    # Embeddings shape: (seq_len, 2304)
    input_dim = 2304
    
    if args.model == 'baseline':
        model = BaselineMLP(input_dim=input_dim)
    elif args.model == 'attention':
        model = AttentionMLP(input_dim=input_dim)
    elif args.model == 'lstm':
        model = BiLSTMClassifier(input_dim=input_dim, lstm_hidden=128)
    else:
        raise ValueError(f"Unknown model architecture: {args.model}")
        
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    epochs_no_improve = 0
    checkpoint_dir = os.path.join(proj_dir, 'models')
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, f"best_{args.model}.pt")
    
    print(f"Starting training for {args.model} model...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(
            model, val_loader, criterion, device
        )
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | "
              f"Train Loss: {train_loss:.4f} - Acc: {train_acc*100:.2f}% - F1: {train_f1*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} - Acc: {val_acc*100:.2f}% - F1: {val_f1*100:.2f}%")
              
        # Checkpoint based on validation F1 score (higher is better)
        if val_f1 > best_val_f1 or (val_f1 == best_val_f1 and val_loss < best_val_loss):
            best_val_loss = val_loss
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_f1': val_f1,
                'model_type': args.model,
                'input_dim': input_dim
            }, best_model_path)
            print(f" => Saved new best checkpoint to {best_model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered after {epoch} epochs. No improvement for {args.patience} epochs.")
                break
                
    print(f"Training completed. Best Validation F1: {best_val_f1*100:.2f}%")

if __name__ == '__main__':
    main()
