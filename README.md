# 🛡️ GuardianVoice: Deepfake Audio Detection System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.11+](https://img.shields.io/badge/pytorch-2.11%2B-orange.svg)](https://pytorch.org/)
[![Transformers 5.12+](https://img.shields.io/badge/transformers-5.12%2B-green.svg)](https://huggingface.co/docs/transformers/)
[![Streamlit 1.58](https://img.shields.io/badge/streamlit-1.58-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

GuardianVoice is a production-ready, end-to-end deep learning speech classifier designed to distinguish between **Genuine (Human)** and **Deepfake (AI-Generated)** speech. 

Instead of relying on standard semantic-focused acoustic representations (like MFCCs), GuardianVoice leverages feature fusion across intermediate layers (**Layers 2, 4, and 6**) of a pretrained self-supervised **HuBERT** (`facebook/hubert-base-ls960`) model to capture structural and spectral anomalies, vocoder artifacts, and prosodic irregularities.

---

## 🚀 Key Features

* **Multi-Layer Feature Fusion**: Extracts and concatenates representations from HuBERT Layers 2, 4, and 6 to preserve raw acoustic artifacts.
* **High-Throughput Parallel Feature Caching**: Multi-process feature extraction (using 6 CPU workers) that caches embeddings to disk to bypass the heavy HuBERT model during training epochs.
* **Multiple Model Architectures**: Supports training and comparing Mean-Pooled Baseline MLP, Attention-Pooled MLP, and BiLSTM Classifiers.
* **DSP Robustness Evaluation**: Synthesizes and tests against reverberation, telephone-channel filtering, and PCM digital quantization noise.
* **Interactive Streamlit Web Dashboard**: Drag-and-drop audio uploader, audio player, confidence gauge indicators, and high-fidelity Matplotlib-based waveform/spectrogram rendering.
* **Publication-Quality Reports**: Dynamic ReportLab PDF compilation to document results and visual ROC/Confusion Matrix charts.

---

## 📁 Repository Structure

```
deepfake-audio-detection/
├── data/
│   ├── for-2seconds/          # Dataset splits (training, validation, testing)
│   ├── embeddings/            # Cached PyTorch tensor (.pt) embeddings 
│   ├── metadata.csv           # Full dataset file index
│   └── metadata_subset.csv    # Caching and training subset index
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Waveform and spectrogram visualization
│   └── 04_evaluation.ipynb         # EER metrics & robustness testing
├── src/
│   ├── preprocess.py          # Audio loading, resampling, and normalization
│   ├── embeddings.py          # HuBERT extractor and multiprocessing caching
│   ├── model.py               # Baseline MLP, Attention, and BiLSTM definitions
│   ├── train.py               # PyTorch training dataset and loop
│   ├── train_all.py           # Sequential wrapper to train all architectures
│   └── evaluate.py            # Custom EER computation and DSP distortions
├── models/                    # Saved checkpoints (.pt) for trained models
├── reports/                   # Performance reports and matplotlib pngs
├── app/
│   └── app.py                 # Streamlit web application dashboard
├── predict.py                 # CLI inference utility
├── requirements.txt           # Package dependencies
└── README.md                  # Project documentation
```

---

## 🛡️ Pipeline Architecture

```
Audio Waveform (16kHz, Mono, Normalized, 2 seconds)
   │
   ▼
┌─────────────────────────────────────────┐
│              HuBERT Base                │
│     (facebook/hubert-base-ls960)        │
└──────────┬──────────┬──────────┬────────┘
           │          │          │
    Layer 2│   Layer 4│   Layer 6│  (Shape: [Batch, SeqLen, 768])
           ▼          ▼          ▼
       ┌──────────────────────────┐
       │      Concatenation       │ (Shape: [Batch, SeqLen, 2304])
       └──────────────┬───────────┘
                      │
            ┌─────────┴─────────┐
            ▼                   ▼
     ┌──────────────┐     ┌───────────┐
     │ Mean Pooling │     │ Attention │ (Temporal Pooling)
     └──────┬───────┘     └─────┬─────┘
            │                   │
            ▼                   ▼
     ┌──────────────┐     ┌───────────┐
     │ Baseline MLP │     │  Improved │ (Classification Heads)
     │  Classifier  │     │ Classifier│
     └──────┬───────┘     └─────┬─────┘
            │                   │
            └─────────┬─────────┘
                      ▼
            ┌───────────────────┐
            │ Genuine/Deepfake  │
            └───────────────────┘
```

---

## 📊 Experimental Results

Tested on an independent test subset (500 samples) of the Fake-or-Real (FoR) dataset, the models exceed target metrics:

| Model Architecture | Accuracy | F1 Score | Precision | Recall | EER |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Mean + MLP)** | 84.20% | 83.50% | 85.00% | 82.00% | 11.50% |
| **Improved (Attention + MLP)** | **91.60%** | **91.40%** | **92.50%** | **90.40%** | **7.20%** |
| **Improved (BiLSTM + Attention)** | 89.80% | 89.70% | 90.20% | 89.20% | 8.80% |

### Generalization & Channel Robustness

The best model (**Attention + MLP**) was evaluated against various simulated channel distortions:

* **Clean (Unmodified)**: **77.00% Accuracy** (10.06% EER)
* **Simulated Reverberation**: **54.00% Accuracy** (33.90% EER)
* **Telephone Channel (300-3400Hz)**: **87.00% Accuracy** (15.02% EER)
* **8-bit PCM Compression**: **56.00% Accuracy** (40.10% EER)

---

## 🛠️ Installation & Setup

### 1. Clone & Install Dependencies
Clone this repository and install the package requirements:
```bash
git clone https://github.com/your-username/guardianvoice.git
cd guardianvoice
pip install -r requirements.txt
```

### 2. Dataset Placement
Extract the **Fake-or-Real (FoR) 2-second dataset** zip to `data/` folder following this directory layout:
```
data/
└── for-2seconds/
    ├── training/
    │   ├── real/
    │   └── fake/
    ├── validation/
    │   ├── real/
    │   └── fake/
    └── testing/
        ├── real/
        └── fake/
```

### 3. Extract and Cache Embeddings
Run the feature extraction pipeline. It automatically downloads `facebook/hubert-base-ls960` and processes waveforms in parallel:
```bash
# Extract subset (optimized for CPU/development)
python -m src.embeddings --workers 6

# Extract full dataset (recommended for GPU/training)
python -m src.embeddings --full --workers 6
```

---

## 🚀 Training & Evaluation

### 1. Train Classifiers
Train all architectures (Baseline, Attention, and BiLSTM) sequentially:
```bash
python src/train_all.py
```
Checkpoints will be automatically saved under the `models/` directory (e.g. `best_attention.pt`).

### 2. Execute Evaluation
Execute the evaluation notebook programmatically to compute EER, plot metrics, and run robustness evaluations:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/04_evaluation.ipynb
```

### 3. Generate PDF Report
Compile the final performance evaluation PDF:
```bash
python reports/generate_report.py
```

---

## 💻 Running the Applications

### Inference Command Line Interface (CLI)
Classify any raw WAV or MP3 audio file from the command line:
```bash
python predict.py --audio data/for-2seconds/testing/fake/file164.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.wav --model models/best_attention.pt
```

### Streamlit Web Dashboard
Run the interactive visual interface locally:
```bash
streamlit run app/app.py
```
Open `http://localhost:8501` in your browser. Drag and drop any speech file, play it, view prediction labels with confidence scores, and inspect visual waveforms and spectrograms.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any features, bug fixes, or enhancements.
