import os
import json

proj_dir = 'C:/Users/Shreshtha Shrinivas/.gemini/antigravity/scratch/deepfake-audio-detection'
notebook_path = os.path.join(proj_dir, 'notebooks', '01_data_exploration.ipynb')

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Phase 1: Dataset Exploration\n",
            "\n",
            "This notebook analyzes the Fake-or-Real (FoR) dataset, focusing on:\n",
            "- Class balance across splits (Training, Validation, Testing)\n",
            "- Audio file durations\n",
            "- Sample rates and channel configurations\n",
            "- Waveform and Mel-spectrogram comparisons between Genuine (Human) and Deepfake (AI-Generated) samples."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import librosa\n",
            "import librosa.display\n",
            "import IPython.display as ipd\n",
            "\n",
            "# Set plot aesthetics\n",
            "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
            "plt.rcParams['figure.figsize'] = (12, 6)\n",
            "plt.rcParams['font.size'] = 12"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Load Metadata\n",
            "\n",
            "We load the generated `metadata.csv` containing information about all files."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "metadata_path = '../data/metadata.csv'\n",
            "df = pd.read_csv(metadata_path)\n",
            "print(f\"Total samples in metadata: {len(df)}\")\n",
            "df.head()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Analyze Class Balance and Splits"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "split_counts = df.groupby(['split', 'label']).size().unstack(fill_value=0)\n",
            "print(\"Sample counts per split and label:\")\n",
            "print(split_counts)\n",
            "\n",
            "# Plot split and label distribution\n",
            "split_counts.plot(kind='bar', stacked=True, color=['#ff6b6b', '#4dadf7'])\n",
            "plt.title('Dataset Split and Class Balance')\n",
            "plt.xlabel('Dataset Split')\n",
            "plt.ylabel('Number of Samples')\n",
            "plt.xticks(rotation=0)\n",
            "plt.legend(['Deepfake (Fake)', 'Genuine (Real)'])\n",
            "plt.tight_layout()\n",
            "plt.savefig('../reports/class_distribution.png', dpi=300)\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Analyze Audio Durations, Sample Rates, and Channels"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"Duration Statistics (seconds):\")\n",
            "print(df['duration'].describe())\n",
            "\n",
            "print(\"\\nUnique Sample Rates:\")\n",
            "print(df['samplerate'].value_counts())\n",
            "\n",
            "print(\"\\nUnique Channel Configurations:\")\n",
            "print(df['channels'].value_counts())"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Visualize Waveforms and Mel Spectrograms\n",
            "\n",
            "Let's select one real sample and one fake sample from the training set, load their waveforms, and plot their waveforms and mel-spectrograms."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "real_sample_path = '../' + df[df['label'] == 'real']['filepath'].iloc[0]\n",
            "fake_sample_path = '../' + df[df['label'] == 'fake']['filepath'].iloc[0]\n",
            "\n",
            "y_real, sr_real = librosa.load(real_sample_path, sr=None)\n",
            "y_fake, sr_fake = librosa.load(fake_sample_path, sr=None)\n",
            "\n",
            "print(f\"Real sample path: {real_sample_path} | Sample Rate: {sr_real} Hz | Shape: {y_real.shape}\")\n",
            "print(f\"Fake sample path: {fake_sample_path} | Sample Rate: {sr_fake} Hz | Shape: {y_fake.shape}\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Plot Waveforms"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 8))\n",
            "\n",
            "librosa.display.waveshow(y_real, sr=sr_real, ax=axes[0], color='#4dadf7')\n",
            "axes[0].set_title('Genuine (Human) Speech Waveform')\n",
            "axes[0].set_ylabel('Amplitude')\n",
            "\n",
            "librosa.display.waveshow(y_fake, sr=sr_fake, ax=axes[1], color='#ff6b6b')\n",
            "axes[1].set_title('Deepfake (AI-Generated) Speech Waveform')\n",
            "axes[1].set_ylabel('Amplitude')\n",
            "axes[1].set_xlabel('Time (seconds)')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig('../reports/waveforms_comparison.png', dpi=300)\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Plot Mel Spectrograms"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "S_real = librosa.feature.melspectrogram(y=y_real, sr=sr_real, n_mels=128, fmax=8000)\n",
            "S_real_db = librosa.power_to_db(S_real, ref=np.max)\n",
            "\n",
            "S_fake = librosa.feature.melspectrogram(y=y_fake, sr=sr_fake, n_mels=128, fmax=8000)\n",
            "S_fake_db = librosa.power_to_db(S_fake, ref=np.max)\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
            "\n",
            "img1 = librosa.display.specshow(S_real_db, x_axis='time', y_axis='mel', sr=sr_real, fmax=8000, ax=axes[0], cmap='viridis')\n",
            "fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')\n",
            "axes[0].set_title('Genuine Mel Spectrogram')\n",
            "\n",
            "img2 = librosa.display.specshow(S_fake_db, x_axis='time', y_axis='mel', sr=sr_fake, fmax=8000, ax=axes[1], cmap='viridis')\n",
            "fig.colorbar(img2, ax=axes[1], format='%+2.0f dB')\n",
            "axes[1].set_title('Deepfake Mel Spectrogram')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.savefig('../reports/spectrograms_comparison.png', dpi=300)\n",
            "plt.show()"
        ]
    }
]

notebook_json = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook_json, f, indent=2)

print("Notebook 01_data_exploration.ipynb created successfully.")
