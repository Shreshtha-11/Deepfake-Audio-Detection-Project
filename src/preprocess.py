import os
import numpy as np
import librosa
import soundfile as sf
import torch

def preprocess_audio(file_path, target_sr=16000, target_duration=2.0):
    """
    Load an audio file, resample, convert to mono, normalize, and pad/truncate to a fixed duration.
    
    Args:
        file_path (str): Path to the audio file.
        target_sr (int): Target sample rate in Hz.
        target_duration (float): Target duration in seconds.
        
    Returns:
        np.ndarray: Preprocessed waveform array of shape (target_sr * target_duration,)
    """
    # Load audio
    # librosa.load resamples to target_sr and converts to mono automatically if sr and mono are specified
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    
    # Normalize amplitude to [-1.0, 1.0] range
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
        
    # Calculate target length in samples
    target_length = int(target_sr * target_duration)
    
    # Pad or truncate
    if len(y) < target_length:
        # Pad with zeros at the end
        pad_width = target_length - len(y)
        y = np.pad(y, (0, pad_width), mode='constant')
    elif len(y) > target_length:
        # Truncate to the first target_length samples
        y = y[:target_length]
        
    return y

def preprocess_audio_tensor(file_path, target_sr=16000, target_duration=2.0):
    """
    Load and preprocess audio, returning it as a PyTorch tensor.
    """
    y = preprocess_audio(file_path, target_sr, target_duration)
    return torch.tensor(y, dtype=torch.float32)

if __name__ == '__main__':
    # Simple verification code
    proj_dir = 'C:/Users/Shreshtha Shrinivas/.gemini/antigravity/scratch/deepfake-audio-detection'
    metadata_path = os.path.join(proj_dir, 'data', 'metadata.csv')
    
    import pandas as pd
    if os.path.exists(metadata_path):
        df = pd.read_csv(metadata_path)
        if len(df) > 0:
            sample_rel = df.iloc[0]['filepath']
            sample_abs = os.path.join(proj_dir, sample_rel)
            print(f"Testing preprocessing on: {sample_abs}")
            waveform = preprocess_audio(sample_abs)
            print(f"Preprocessed waveform shape: {waveform.shape}")
            print(f"Min value: {waveform.min():.4f}, Max value: {waveform.max():.4f}")
            print(f"Sample Rate: 16000 (Target), Duration: {len(waveform)/16000:.2f} seconds")
            assert len(waveform) == 32000, "Waveform length must be exactly 32000 samples (2s @ 16kHz)"
            print("Preprocessing test passed!")
        else:
            print("Metadata is empty.")
    else:
        print("Metadata not found.")
