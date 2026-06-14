import os
import json

proj_dir = 'C:/Users/Shreshtha Shrinivas/.gemini/antigravity/scratch/deepfake-audio-detection'
notebook_path = os.path.join(proj_dir, 'notebooks', '04_evaluation.ipynb')

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Phase 7 & 8: Model Evaluation and Generalization\n",
            "\n",
            "This notebook evaluates the trained deepfake audio detection classifiers on the testing set. It computes standard classification metrics, details the Equal Error Rate (EER), and measures model robustness against simulated environmental and channel distortions."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "sys.path.insert(0, os.path.abspath('..'))\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import torch\n",
            "from torch.utils.data import DataLoader\n",
            "\n",
            "from sklearn.metrics import accuracy_score\n",
            "from src.train import EmbeddingDataset\n",
            "from src.evaluate import evaluate_model, plot_confusion_matrix, plot_roc_curve, calculate_eer\n",
            "from src.evaluate import apply_reverberation, apply_bandpass_filter, apply_quantization\n",
            "from src.preprocess import preprocess_audio_tensor, preprocess_audio\n",
            "from src.embeddings import HubertExtractor\n",
            "from src.model import BaselineMLP, AttentionMLP, BiLSTMClassifier\n",
            "\n",
            "# Aesthetics\n",
            "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
            "plt.rcParams['figure.figsize'] = (10, 6)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Load Test Dataset and Models\n",
            "\n",
            "We instantiate the test loader and load the checkpoints for our trained classifiers."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
            "proj_dir = 'C:/Users/Shreshtha Shrinivas/.gemini/antigravity/scratch/deepfake-audio-detection'\n",
            "metadata_path = os.path.join(proj_dir, 'data', 'metadata_subset.csv')\n",
            "\n",
            "test_dataset = EmbeddingDataset(metadata_path, 'testing', proj_dir)\n",
            "test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)\n",
            "print(f\"Loaded test set with {len(test_dataset)} samples.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Evaluate Classifiers"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def load_checkpoint(model_type):\n",
            "    checkpoint_path = os.path.join(proj_dir, 'models', f'best_{model_type}.pt')\n",
            "    if not os.path.exists(checkpoint_path):\n",
            "        print(f\"Warning: Checkpoint not found for {model_type}\")\n",
            "        return None\n",
            "    checkpoint = torch.load(checkpoint_path, map_location=device)\n",
            "    input_dim = checkpoint.get('input_dim', 2304)\n",
            "    \n",
            "    if model_type == 'baseline':\n",
            "        model = BaselineMLP(input_dim=input_dim)\n",
            "    elif model_type == 'attention':\n",
            "        model = AttentionMLP(input_dim=input_dim)\n",
            "    elif model_type == 'lstm':\n",
            "        model = BiLSTMClassifier(input_dim=input_dim, lstm_hidden=128)\n",
            "        \n",
            "    model.load_state_dict(checkpoint['model_state_dict'])\n",
            "    model.to(device)\n",
            "    return model\n",
            "\n",
            "models = {}\n",
            "for m_type in ['baseline', 'attention', 'lstm']:\n",
            "    loaded = load_checkpoint(m_type)\n",
            "    if loaded:\n",
            "        models[m_type] = loaded"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "results = []\n",
            "eval_data = {}\n",
            "\n",
            "for name, model in models.items():\n",
            "    print(f\"Evaluating {name}...\")\n",
            "    metrics, y_true, y_pred, y_scores = evaluate_model(model, test_loader, device)\n",
            "    \n",
            "    # Save predictions for plots\n",
            "    eval_data[name] = {\n",
            "        'metrics': metrics,\n",
            "        'y_true': y_true,\n",
            "        'y_pred': y_pred,\n",
            "        'y_scores': y_scores\n",
            "    }\n",
            "    \n",
            "    results.append({\n",
            "        'Model': name.upper(),\n",
            "        'Accuracy': f\"{metrics['accuracy']*100:.2f}%\",\n",
            "        'F1 Score': f\"{metrics['f1']*100:.2f}%\",\n",
            "        'Precision': f\"{metrics['precision']*100:.2f}%\",\n",
            "        'Recall': f\"{metrics['recall']*100:.2f}%\",\n",
            "        'EER': f\"{metrics['eer']*100:.2f}%\",\n",
            "        'Per-Class Real Acc': f\"{metrics['real_accuracy']*100:.2f}%\",\n",
            "        'Per-Class Fake Acc': f\"{metrics['fake_accuracy']*100:.2f}%\"\n",
            "    })\n",
            "\n",
            "df_results = pd.DataFrame(results)\n",
            "df_results"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Generate Evaluation Figures for Best Model\n",
            "\n",
            "We pick the model with the lowest Equal Error Rate (EER) and plot its ROC Curve and Confusion Matrix."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Select best model type\n",
            "if eval_data:\n",
            "    best_model_name = min(eval_data.keys(), key=lambda k: eval_data[k]['metrics']['eer'])\n",
            "    print(f\"Best model by EER: {best_model_name.upper()}\")\n",
            "    \n",
            "    best_metrics = eval_data[best_model_name]['metrics']\n",
            "    best_y_true = eval_data[best_model_name]['y_true']\n",
            "    best_y_scores = eval_data[best_model_name]['y_scores']\n",
            "    best_y_pred = eval_data[best_model_name]['y_pred']\n",
            "    \n",
            "    # Generate and save Confusion Matrix\n",
            "    cm_path = os.path.join(proj_dir, 'reports', 'confusion_matrix.png')\n",
            "    plot_confusion_matrix(best_metrics['confusion_matrix'], cm_path)\n",
            "    print(f\"Confusion matrix saved to {cm_path}\")\n",
            "    \n",
            "    # Generate and save ROC Curve\n",
            "    roc_path = os.path.join(proj_dir, 'reports', 'roc_curve.png')\n",
            "    plot_roc_curve(best_y_true, best_y_scores, best_metrics['eer'], roc_path)\n",
            "    print(f\"ROC curve saved to {roc_path}\")\n",
            "else:\n",
            "    print(\"No models evaluated.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Phase 8: Robustness and Channel Distortion Simulation\n",
            "\n",
            "To measure how our model generalizes, we simulate physical rerecording, channel bandpass filtering, and PCM digital compression."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# We perform evaluation by processing audio waveforms directly\n",
            "# We load the test subset metadata, load and transform each audio, run it through HuBERT on the fly, and then feed it into the best classifier model\n",
            "df_test = pd.read_csv(metadata_path)\n",
            "df_test = df_test[df_test['split'] == 'testing'].reset_index(drop=True)\n",
            "\n",
            "# Limit to 100 samples for fast evaluation on CPU\n",
            "df_test_sample = df_test.sample(n=min(100, len(df_test)), random_state=42).reset_index(drop=True)\n",
            "print(f\"Evaluating channel distortion on balanced sample size: {len(df_test_sample)}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load HuBERT extractor\n",
            "extractor = HubertExtractor(device=device)\n",
            "best_classifier = models[best_model_name]\n",
            "best_classifier.eval()\n",
            "\n",
            "def run_robustness_eval(transform_fn=None, name=\"Clean\"):\n",
            "    all_labels = []\n",
            "    all_scores = []\n",
            "    all_preds = []\n",
            "    \n",
            "    label_map = {'real': 0, 'fake': 1}\n",
            "    \n",
            "    for _, row in df_test_sample.iterrows():\n",
            "        file_path = os.path.join(proj_dir, row['filepath'])\n",
            "        label = label_map[row['label']]\n",
            "        \n",
            "        # 1. Preprocess\n",
            "        waveform = preprocess_audio(file_path)\n",
            "        \n",
            "        # 2. Apply distortion\n",
            "        if transform_fn is not None:\n",
            "            waveform = transform_fn(waveform)\n",
            "            \n",
            "        # 3. Extract embeddings\n",
            "        waveform_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(device)\n",
            "        with torch.no_grad():\n",
            "            feats = extractor.extract_features(waveform_tensor)\n",
            "            embedding = feats['combined'].unsqueeze(0).to(device)\n",
            "            logits = best_classifier(embedding)\n",
            "            probs = torch.softmax(logits, dim=1)\n",
            "            \n",
            "        all_labels.append(label)\n",
            "        all_scores.append(probs[0, 1].item())\n",
            "        all_preds.append(torch.argmax(logits, dim=1).item())\n",
            "        \n",
            "    y_true = np.array(all_labels)\n",
            "    y_scores = np.array(all_scores)\n",
            "    y_pred = np.array(all_preds)\n",
            "    \n",
            "    acc = accuracy_score(y_true, y_pred)\n",
            "    eer, _ = calculate_eer(y_true, y_scores)\n",
            "    \n",
            "    return acc, eer\n",
            "\n",
            "print(\"Running robustness simulations...\")\n",
            "acc_clean, eer_clean = run_robustness_eval(None, \"Clean\")\n",
            "print(f\"Clean: Acc = {acc_clean*100:.2f}%, EER = {eer_clean*100:.2f}%\")\n",
            "\n",
            "acc_reverb, eer_reverb = run_robustness_eval(apply_reverberation, \"Reverb\")\n",
            "print(f\"Reverb: Acc = {acc_reverb*100:.2f}%, EER = {eer_reverb*100:.2f}%\")\n",
            "\n",
            "acc_tele, eer_tele = run_robustness_eval(lambda w: apply_bandpass_filter(w, lowcut=300, highcut=3400), \"Telephone\")\n",
            "print(f\"Telephone Filter: Acc = {acc_tele*100:.2f}%, EER = {eer_tele*100:.2f}%\")\n",
            "\n",
            "acc_quant, eer_quant = run_robustness_eval(lambda w: apply_quantization(w, bits=8), \"8-bit PCM\")\n",
            "print(f\"8-bit PCM Compression: Acc = {acc_quant*100:.2f}%, EER = {eer_quant*100:.2f}%\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Summary of Distortion Robustness"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "robust_results = [\n",
            "    {'Channel Condition': 'Clean (Unmodified)', 'Accuracy': f'{acc_clean*100:.2f}%', 'EER': f'{eer_clean*100:.2f}%'},\n",
            "    {'Channel Condition': 'Simulated Reverberation', 'Accuracy': f'{acc_reverb*100:.2f}%', 'EER': f'{eer_reverb*100:.2f}%'},\n",
            "    {'Channel Condition': 'Telephone Bandpass Filter (300-3400Hz)', 'Accuracy': f'{acc_tele*100:.2f}%', 'EER': f'{eer_tele*100:.2f}%'},\n",
            "    {'Channel Condition': '8-bit PCM Compression', 'Accuracy': f'{acc_quant*100:.2f}%', 'EER': f'{eer_quant*100:.2f}%'}\n",
            "]\n",
            "df_robust = pd.DataFrame(robust_results)\n",
            "df_robust.to_csv(os.path.join(proj_dir, 'reports', 'robustness_results.csv'), index=False)\n",
            "df_robust"
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

print("Notebook 04_evaluation.ipynb created successfully.")
