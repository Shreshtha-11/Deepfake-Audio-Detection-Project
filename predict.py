import os
import argparse
import torch
import numpy as np

from src.preprocess import preprocess_audio_tensor
from src.embeddings import HubertExtractor
from src.model import BaselineMLP, AttentionMLP, BiLSTMClassifier

def load_model(model_path, device):
    """
    Load a trained classification model and its configuration from a checkpoint.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        
    print(f"Loading classifier checkpoint from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    model_type = checkpoint.get('model_type', 'baseline')
    input_dim = checkpoint.get('input_dim', 2304)
    
    # Instantiate correct model class
    if model_type == 'baseline':
        model = BaselineMLP(input_dim=input_dim)
    elif model_type == 'attention':
        model = AttentionMLP(input_dim=input_dim)
    elif model_type == 'lstm':
        model = BiLSTMClassifier(input_dim=input_dim, lstm_hidden=128)
    else:
        raise ValueError(f"Unknown model type in checkpoint: {model_type}")
        
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, model_type

def main():
    parser = argparse.ArgumentParser(description="Inference CLI for Deepfake Audio Detection")
    parser.add_argument('--audio', type=str, required=True, help="Path to the input audio file (.wav or .mp3)")
    parser.add_argument('--model', type=str, default='models/best_attention.pt', help="Path to the trained model checkpoint")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if files exist
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found at '{args.audio}'")
        return
        
    # Load model first to make sure it exists
    try:
        model, model_type = load_model(args.model, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    # Load HuBERT extractor
    try:
        extractor = HubertExtractor(device=device)
    except Exception as e:
        print(f"Error loading HuBERT extractor: {e}")
        return
        
    # Preprocess audio
    print(f"Preprocessing audio: {args.audio}...")
    try:
        waveform = preprocess_audio_tensor(args.audio)
    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        return
        
    # Extract HuBERT embeddings
    print("Extracting intermediate layers [2, 4, 6] representations...")
    try:
        feats = extractor.extract_features(waveform)
        # Add batch dimension: (1, seq_len, 2304)
        embedding = feats['combined'].unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return
        
    # Run classification
    print("Running classification...")
    with torch.no_grad():
        logits = model(embedding)
        probs = torch.softmax(logits, dim=1)
        
    prob_real = probs[0, 0].item()
    prob_fake = probs[0, 1].item()
    
    if prob_fake > prob_real:
        prediction = "Deepfake"
        confidence = prob_fake * 100
    else:
        prediction = "Genuine"
        confidence = prob_real * 100
        
    print("\n" + "="*40)
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"(Details - Genuine: {prob_real*100:.1f}%, Deepfake: {prob_fake*100:.1f}%)")
    print("="*40)

if __name__ == '__main__':
    main()
