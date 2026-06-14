import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import HubertModel
from src.preprocess import preprocess_audio_tensor
from concurrent.futures import ProcessPoolExecutor

class HubertExtractor:
    def __init__(self, model_name="facebook/hubert-base-ls960", device="cpu"):
        self.device = torch.device(device)
        print(f"Loading HuBERT model '{model_name}' on {self.device}...")
        self.model = HubertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
    def extract_features(self, waveform):
        """
        Extract hidden states from Layer 2, 4, 6 and combine them.
        """
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
            
        waveform = waveform.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(waveform, output_hidden_states=True)
            feat2 = outputs.hidden_states[2].squeeze(0).cpu()  # (seq_len, 768)
            feat4 = outputs.hidden_states[4].squeeze(0).cpu()  # (seq_len, 768)
            feat6 = outputs.hidden_states[6].squeeze(0).cpu()  # (seq_len, 768)
            combined = torch.cat([feat2, feat4, feat6], dim=-1)  # (seq_len, 2304)
            
        return {
            'layer_2': feat2,
            'layer_4': feat4,
            'layer_6': feat6,
            'combined': combined
        }

def get_layer_from_combined(combined_tensor, layer_num):
    if layer_num == 2:
        return combined_tensor[..., 0:768]
    elif layer_num == 4:
        return combined_tensor[..., 768:1536]
    elif layer_num == 6:
        return combined_tensor[..., 1536:2304]
    else:
        raise ValueError("Only layers 2, 4, and 6 are saved in the combined tensor.")

# Multiprocessing Worker Globals and Functions
_extractor = None

def init_worker():
    global _extractor
    import torch
    # Set PyTorch thread count to 1 inside workers to prevent core thrashing
    torch.set_num_threads(1)
    device = "cpu"  # Keep CPU-only as checked
    _extractor = HubertExtractor(device=device)

def process_single_task(task):
    global _extractor
    file_abs, out_file = task
    
    if os.path.exists(out_file):
        return True
        
    try:
        waveform = preprocess_audio_tensor(file_abs)
        feats = _extractor.extract_features(waveform)
        torch.save(feats['combined'], out_file)
        return True
    except Exception as e:
        print(f"Error processing {file_abs}: {e}")
        return False

def cache_embeddings(subset_size=None, num_workers=4):
    """
    Cache embeddings for the dataset in parallel.
    """
    proj_dir = 'C:/Users/Shreshtha Shrinivas/.gemini/antigravity/scratch/deepfake-audio-detection'
    metadata_path = os.path.join(proj_dir, 'data', 'metadata.csv')
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}. Run Phase 1 first.")
        
    df = pd.read_csv(metadata_path)
    
    selected_df = pd.DataFrame()
    if subset_size is not None:
        print(f"Creating a balanced subset of size: {subset_size}")
        for split, count in subset_size.items():
            for label in ['real', 'fake']:
                sub_df = df[(df['split'] == split) & (df['label'] == label)]
                sample_count = min(count, len(sub_df))
                sampled = sub_df.sample(n=sample_count, random_state=42)
                selected_df = pd.concat([selected_df, sampled])
    else:
        selected_df = df
        
    selected_df = selected_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Save the subset metadata
    subset_metadata_path = os.path.join(proj_dir, 'data', 'metadata_subset.csv')
    selected_df.to_csv(subset_metadata_path, index=False)
    print(f"Subset metadata saved to {subset_metadata_path} ({len(selected_df)} files total).")
    
    # Build the task list
    tasks = []
    for idx, row in selected_df.iterrows():
        file_rel = row['filepath']
        file_abs = os.path.join(proj_dir, file_rel)
        label = row['label']
        split = row['split']
        
        # Prepare output directories
        out_dir = os.path.join(proj_dir, 'data', 'embeddings', split, label)
        os.makedirs(out_dir, exist_ok=True)
        
        # Output file name: replace .wav with .pt
        base_name = os.path.splitext(os.path.basename(file_abs))[0]
        out_file = os.path.join(out_dir, f"{base_name}.pt")
        
        tasks.append((file_abs, out_file))
        
    # Execute extraction in parallel
    print(f"Starting parallel extraction with {num_workers} workers...")
    
    success_count = 0
    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker) as executor:
        # We wrap the results with tqdm to monitor progress
        results = list(tqdm(executor.map(process_single_task, tasks), total=len(tasks)))
        success_count = sum(1 for r in results if r)
        
    print(f"Embedding extraction complete! Successfully cached {success_count}/{len(tasks)} files.")

if __name__ == '__main__':
    subset_config = {
        'training': 300,
        'validation': 100,
        'testing': 100
    }
    
    # Use 4 workers by default for high throughput without memory exhaustion
    # Spawning 4 workers takes ~4 * 360MB = 1.4GB RAM, which is very safe.
    import sys
    num_w = 4
    if '--workers' in sys.argv:
        idx = sys.argv.index('--workers')
        num_w = int(sys.argv[idx + 1])
        
    if '--full' in sys.argv:
        cache_embeddings(subset_size=None, num_workers=num_w)
    else:
        cache_embeddings(subset_size=subset_config, num_workers=num_w)
