import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import torch

# Set streamlit page configurations
st.set_page_config(
    page_title="GuardianVoice - Deepfake Audio Detection",
    page_icon="🛡️",
    layout="wide"
)

# Custom premium CSS injection for a dark theme with glassmorphism and gradient highlights
st.markdown("""
<style>
    /* Dark glassmorphic container style */
    .stApp {
        background-color: #0d0f12;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header gradients */
    .title-gradient {
        background: linear-gradient(135deg, #4dadf7 0%, #ff6b6b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .subtitle-text {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Sleek card container */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 24px;
        backdrop-filter: blur(12px);
        margin-bottom: 1.5rem;
    }
    
    /* Result card layouts */
    .result-fake {
        border-left: 5px solid #ff6b6b;
        background: rgba(255, 107, 107, 0.08);
        border-radius: 8px;
        padding: 20px;
        margin-top: 1rem;
    }
    
    .result-real {
        border-left: 5px solid #4dadf7;
        background: rgba(77, 173, 247, 0.08);
        border-radius: 8px;
        padding: 20px;
        margin-top: 1rem;
    }
    
    .result-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    
    .result-fake .result-title {
        color: #ff6b6b;
    }
    
    .result-real .result-title {
        color: #4dadf7;
    }
</style>
""", unsafe_allow_html=True)

# Import local project modules safely
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import preprocess_audio_tensor, preprocess_audio
from src.embeddings import HubertExtractor
from src.model import BaselineMLP, AttentionMLP, BiLSTMClassifier

# Lazy loading model and extractor for caching
@st.cache_resource
def load_classifier(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    model_type = checkpoint.get('model_type', 'baseline')
    input_dim = checkpoint.get('input_dim', 2304)
    
    if model_type == 'baseline':
        model = BaselineMLP(input_dim=input_dim)
    elif model_type == 'attention':
        model = AttentionMLP(input_dim=input_dim)
    elif model_type == 'lstm':
        model = BiLSTMClassifier(input_dim=input_dim, lstm_hidden=128)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, device

@st.cache_resource
def get_extractor():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return HubertExtractor(device=device)

# App UI Header
st.markdown('<div class="title-gradient">GuardianVoice</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Next-Generation Deepfake Speech Detection System powered by HuBERT Representation Fusion</div>', unsafe_allow_html=True)

# Check model availability
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(proj_dir, 'models')
model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')] if os.path.exists(model_dir) else []

if not model_files:
    st.warning("⚠️ No trained models found in the `models/` directory. Please run the training pipeline first.")
    # Show dummy prediction options for presentation/dev fallback
    selected_model_file = None
else:
    selected_model_file = os.path.join(model_dir, model_files[0]) # Default to first found model

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/nolan/256/microphone.png", width=100)
    st.title("Settings")
    st.markdown("---")
    
    if model_files:
        chosen_file = st.selectbox("Select Classifier Model", model_files)
        selected_model_file = os.path.join(model_dir, chosen_file)
        
        # Display details from model checkpoint
        checkpoint_data = torch.load(selected_model_file, map_location='cpu')
        st.info(f"**Model Type:** {checkpoint_data.get('model_type', 'N/A').upper()}\n\n"
                f"**Val F1:** {checkpoint_data.get('val_f1', 0.0)*100:.2f}%\n\n"
                f"**Val Loss:** {checkpoint_data.get('val_loss', 0.0):.4f}")
    else:
        st.error("Please run the training pipeline to generate models.")

# Main Application Layout - Grid Columns
col1, col2 = st.columns([1, 1.2])

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Upload Audio Sample")
    
    uploaded_file = st.file_uploader(
        "Choose an audio file (WAV or MP3 format)", 
        type=["wav", "mp3"], 
        help="Upload the audio sample you want to analyze for synthetic voice artifacts."
    )
    
    if uploaded_file is not None:
        # Save file temporarily to disk for preprocessing
        temp_dir = os.path.join(proj_dir, 'data', 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.audio(temp_file_path, format="audio/wav")
        
        # Analyze button
        analyze_btn = st.button("🛡️ Run Deepfake Analysis", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if analyze_btn and selected_model_file:
            with st.spinner("Analyzing acoustic features..."):
                try:
                    # 1. Load model and extractor
                    model, device = load_classifier(selected_model_file)
                    extractor = get_extractor()
                    
                    # 2. Preprocess audio
                    waveform_tensor = preprocess_audio_tensor(temp_file_path)
                    
                    # 3. Extract embeddings
                    feats = extractor.extract_features(waveform_tensor)
                    # Add batch dim: (1, seq_len, 2304)
                    embedding = feats['combined'].unsqueeze(0).to(device)
                    
                    # 4. Model Inference
                    with torch.no_grad():
                        logits = model(embedding)
                        probs = torch.softmax(logits, dim=1)
                        
                    prob_real = probs[0, 0].item()
                    prob_fake = probs[0, 1].item()
                    
                    # Store variables in session state to display on the right column
                    st.session_state['analysis_done'] = True
                    st.session_state['prob_real'] = prob_real
                    st.session_state['prob_fake'] = prob_fake
                    st.session_state['temp_file'] = temp_file_path
                    
                except Exception as e:
                    st.error(f"Error during audio processing: {e}")
    else:
        st.info("💡 Please upload an audio file to begin analysis.")
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Analysis Results & Visualizations")
    
    if st.session_state.get('analysis_done', False):
        prob_real = st.session_state['prob_real']
        prob_fake = st.session_state['prob_fake']
        temp_file = st.session_state['temp_file']
        
        # Results Display Card
        if prob_fake > prob_real:
            st.markdown(f"""
            <div class="result-fake">
                <div class="result-title">🚨 DEEPFAKE DETECTED</div>
                <div>The audio sample shows clear spectral and vocoder anomalies matching synthetic, AI-generated speech.</div>
                <div style="font-weight: 700; margin-top: 10px; font-size: 1.2rem;">Confidence Score: {prob_fake*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-real">
                <div class="result-title">✅ GENUINE SPEECH</div>
                <div>The audio sample matches natural human speech characteristics with normal prosody and vocal tract resonances.</div>
                <div style="font-weight: 700; margin-top: 10px; font-size: 1.2rem;">Confidence Score: {prob_real*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Probability Bar Chart
        st.markdown("<br>", unsafe_allow_html=True)
        st.write("**Classification Probabilities**")
        
        # Display custom styled progress bars
        st.write("Genuine Speech (Human)")
        st.progress(prob_real)
        st.write("Deepfake Speech (AI)")
        st.progress(prob_fake)
        
        # Waveform and Mel Spectrogram plots
        st.markdown("<br>", unsafe_allow_html=True)
        st.write("**Visual Audio Signatures**")
        
        # Load audio for plotting
        y, sr = librosa.load(temp_file, sr=16000, mono=True)
        
        # Draw plots
        fig, axes = plt.subplots(2, 1, figsize=(10, 6.5))
        
        # Waveform Plot
        librosa.display.waveshow(y, sr=sr, ax=axes[0], color='#4dadf7' if prob_real > prob_fake else '#ff6b6b')
        axes[0].set_title('Time-Domain Audio Waveform', color='white', fontsize=11)
        axes[0].set_ylabel('Amplitude', color='white')
        axes[0].tick_params(colors='gray')
        axes[0].grid(True, linestyle=':', alpha=0.3)
        
        # Mel Spectrogram Plot
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_db = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=axes[1], cmap='viridis')
        axes[1].set_title('Frequency-Domain Mel Spectrogram', color='white', fontsize=11)
        axes[1].set_ylabel('Mel Frequency', color='white')
        axes[1].tick_params(colors='gray')
        
        # Style layout for dark mode
        fig.patch.set_facecolor('#1e293b')
        for ax in axes:
            ax.set_facecolor('#0d0f12')
            ax.title.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            
        plt.tight_layout()
        st.pyplot(fig)
        
    else:
        st.info("Run analysis on the uploaded file to view detailed audio signatures and classification probabilities.")
        
    st.markdown('</div>', unsafe_allow_html=True)
