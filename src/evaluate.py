import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve
import seaborn as sns
from scipy import signal

def calculate_eer(y_true, y_scores):
    """
    Calculate the Equal Error Rate (EER) and the corresponding threshold.
    
    Args:
        y_true (np.ndarray): Binary labels (0 = real, 1 = fake).
        y_scores (np.ndarray): Prediction probabilities/scores for the positive class (fake).
        
    Returns:
        float: EER value (as a fraction, e.g. 0.08 for 8%).
        float: Threshold where FAR and FRR are closest.
    """
    # Compute False Positive Rate (FPR / FAR) and True Positive Rate (TPR)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    
    # False Negative Rate (FNR / FRR) = 1 - TPR
    fnr = 1 - tpr
    
    # EER is where FPR equals FNR. Find the threshold index where |FPR - FNR| is minimized
    idx = np.nanargmin(np.absolute(fpr - fnr))
    
    # We can average fpr and fnr at that point for stability, or just use fpr[idx]
    eer = (fpr[idx] + fnr[idx]) / 2
    threshold = thresholds[idx]
    
    return eer, threshold

def evaluate_model(model, dataloader, device):
    """
    Runs model on dataset, computes metrics, and returns predictions.
    """
    model.eval()
    all_labels = []
    all_logits = []
    all_scores = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            logits = model(inputs)
            
            # Compute probabilities (softmax)
            probs = torch.softmax(logits, dim=1)
            
            all_labels.extend(labels.numpy())
            all_logits.extend(logits.cpu().numpy())
            all_scores.extend(probs[:, 1].cpu().numpy())  # Probability of fake (class 1)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_scores = np.array(all_scores)
    
    # Standard metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    # Calculate EER
    eer, eer_threshold = calculate_eer(y_true, y_scores)
    
    # Per-class accuracy
    cm = confusion_matrix(y_true, y_pred)
    # cm format: [[TN, FP], [FN, TP]] where 0 is real, 1 is fake
    # Genuine class accuracy = TN / (TN + FP)
    # Deepfake class accuracy = TP / (FN + TP)
    tn, fp, fn, tp = cm.ravel()
    real_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
    fake_acc = tp / (fn + tp) if (fn + tp) > 0 else 0
    
    metrics = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'eer': eer,
        'eer_threshold': eer_threshold,
        'real_accuracy': real_acc,
        'fake_accuracy': fake_acc,
        'confusion_matrix': cm
    }
    
    return metrics, y_true, y_pred, y_scores

def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Genuine', 'Deepfake'],
        yticklabels=['Genuine', 'Deepfake']
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_curve(y_true, y_scores, eer, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#4dadf7', label=f'ROC Curve (EER = {eer*100:.2f}%)')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.plot([eer, eer], [0, 1 - eer], color='#ff6b6b', linestyle=':', label='Equal Error Rate Line')
    plt.scatter([eer], [1 - eer], color='#ff6b6b', marker='o', s=100, zorder=5)
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# DSP Transformations for Robustness Simulation (Phase 8)

def apply_reverberation(waveform, sr=16000):
    """
    Simulate reverberation using a multi-tap delay line.
    """
    # Simple multi-tap feedback delay line mimicking a small room
    # delay times: 15ms, 30ms, 45ms
    delays = [int(0.015 * sr), int(0.030 * sr), int(0.045 * sr)]
    gains = [0.4, 0.25, 0.15]
    
    reverb = np.copy(waveform)
    for d, g in zip(delays, gains):
        delayed = np.zeros_like(waveform)
        delayed[d:] = waveform[:-d]
        reverb += g * delayed
        
    # Normalize
    max_val = np.max(np.abs(reverb))
    if max_val > 0:
        reverb = reverb / max_val
    return reverb

def apply_bandpass_filter(waveform, sr=16000, lowcut=300, highcut=3400):
    """
    Simulate telephone bandpass filtering (300Hz to 3400Hz).
    """
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    
    # 4th order Butterworth bandpass filter
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.lfilter(b, a, waveform)
    
    # Normalize
    max_val = np.max(np.abs(filtered))
    if max_val > 0:
        filtered = filtered / max_val
    return filtered

def apply_quantization(waveform, bits=8):
    """
    Simulate digital quantization noise / compression (e.g. 8-bit PCM).
    """
    levels = 2 ** bits
    # Map [-1.0, 1.0] to [0, levels - 1]
    quantized = np.round((waveform + 1.0) / 2.0 * (levels - 1))
    # Map back to [-1.0, 1.0]
    quantized = (quantized / (levels - 1) * 2.0) - 1.0
    return quantized
