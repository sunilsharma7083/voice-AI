import torch
import torchaudio
import faiss
import numpy as np
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model
)

# =========================================================
# GLOBAL CONFIG
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000


# =========================================================
# 1. EAR: Whisper (Speech-to-Text with Language Detection)
# =========================================================

class WhisperASR(torch.nn.Module):
    def __init__(self, model_name="openai/whisper-large-v3"):
        super().__init__()
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def forward(self, waveform):
        # Ensure numpy audio
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.squeeze().cpu().numpy()

        inputs = self.processor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )

        input_features = inputs.input_features.to(DEVICE)

        # ✅ CORRECT language detection (NO generate, NO decode)
        lang_probs = self.processor.tokenizer.detect_language(input_features)
        detected_lang = max(lang_probs[0], key=lang_probs[0].get)
        # detected_lang will be: "english", "hindi", "tamil", etc.

        # ✅ Correct transcription with detected language
        predicted_ids = self.model.generate(
            input_features,
            task="transcribe",
            language=detected_lang
        )

        text = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]

        return text, detected_lang.capitalize()


# =========================================================
# 2. BRAIN-A: Speaker Embeddings (Wav2Vec2 / XLS-R)
# =========================================================

class SpeakerEncoder(torch.nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-large-xlsr-53"):
        super().__init__()
        # 🔑 Use FeatureExtractor instead of Processor (no tokenizer needed)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def forward(self, waveform):
        inputs = self.processor(
            waveform.squeeze().cpu().numpy(),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        ).to(DEVICE)

        hidden_states = self.model(**inputs).last_hidden_state
        embedding = hidden_states.mean(dim=1)
        return torch.nn.functional.normalize(embedding, dim=-1)


# =========================================================
# 2. BRAIN-B: Acoustic Features (Enhanced with Age Detection)
# =========================================================

class AcousticFeatureExtractor(torch.nn.Module):
    def forward(self, waveform):
        # Basic acoustic features
        pitch = torchaudio.functional.detect_pitch_frequency(waveform, SAMPLE_RATE)
        energy = waveform.pow(2).mean(dim=-1)
        zcr = ((waveform[:, 1:] * waveform[:, :-1]) < 0).float().mean(dim=-1)
        
        # Spectral features for age estimation
        spectrogram = torchaudio.transforms.Spectrogram()(waveform)
        spectral_centroid = self.compute_spectral_centroid(spectrogram)
        
        # Formant-like features (approximation)
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_mels=40
        )(waveform)
        
        features = torch.stack([
            pitch.mean(),
            energy.mean(),
            zcr.mean(),
            spectral_centroid
        ])

        return features.to(DEVICE), mel_spec.to(DEVICE)
    
    def compute_spectral_centroid(self, spectrogram):
        """Compute spectral centroid - useful for age estimation"""
        magnitude = spectrogram.abs()
        freqs = torch.linspace(0, SAMPLE_RATE / 2, magnitude.shape[-2]).to(spectrogram.device)
        centroid = (magnitude * freqs.unsqueeze(-1)).sum(dim=-2) / (magnitude.sum(dim=-2) + 1e-8)
        return centroid.mean()


# =========================================================
# 3. AGE ESTIMATOR (Based on Acoustic Features)
# =========================================================

class AgeEstimator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Simple MLP for age estimation based on acoustic features
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(4, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(32, 8)  # 8 age groups
        ).to(DEVICE)
        
        self.age_groups = [
            "Infant (0-3)",
            "Child (4-12)", 
            "Teenager (13-17)",
            "Young Adult (18-25)",
            "Adult (26-40)",
            "Middle-Aged (41-55)",
            "Senior (56-70)",
            "Elderly (71+)"
        ]
    
    @torch.no_grad()
    def forward(self, acoustic_features):
        """Estimate age group from acoustic features"""
        # Rule-based age estimation (heuristic approach)
        pitch_mean = acoustic_features[0].item()
        energy_mean = acoustic_features[1].item()
        zcr_mean = acoustic_features[2].item()
        spectral_centroid = acoustic_features[3].item()
        
        # Enhanced heuristic rules for age estimation
        # Based on acoustic characteristics that change with age
        
        # Very high pitch + high spectral features = Infant/very young child
        if pitch_mean > 300 and spectral_centroid > 2500:
            age_group = "Infant (0-3)"
        
        # High pitch (typical of children)
        elif pitch_mean > 250 and spectral_centroid > 2200:
            age_group = "Child (4-12)"
        
        # Moderately high pitch with high energy (teenagers going through voice changes)
        elif pitch_mean > 200 and energy_mean > 0.01 and spectral_centroid > 2000:
            age_group = "Teenager (13-17)"
        
        # Clear, energetic voice (young adults)
        elif pitch_mean > 160 and spectral_centroid > 1800 and energy_mean > 0.008:
            age_group = "Young Adult (18-25)"
        
        # Stable voice characteristics (adults)
        elif pitch_mean > 130 and spectral_centroid > 1500:
            age_group = "Adult (26-40)"
        
        # Slightly lower pitch, stable energy (middle-aged)
        elif pitch_mean > 110 and spectral_centroid > 1200:
            age_group = "Middle-Aged (41-55)"
        
        # Lower pitch, potentially reduced energy (seniors)
        elif pitch_mean > 90 or (energy_mean > 0.005 and spectral_centroid > 1000):
            age_group = "Senior (56-70)"
        
        # Lowest pitch and energy characteristics (elderly)
        else:
            age_group = "Elderly (71+)"
        
        return age_group


# =========================================================
# 4. MEMORY: FAISS (Cosine Similarity)
# =========================================================

class VoiceMemory:
    def __init__(self, dim):
        self.index = faiss.IndexFlatIP(dim)  # Cosine similarity
        self.metadata = []

    def add(self, embedding, meta):
        emb = embedding.detach().cpu().numpy()
        faiss.normalize_L2(emb)
        self.index.add(emb)
        self.metadata.append(meta)

    def search(self, embedding, k=3):
        if self.index.ntotal == 0:
            return []
        emb = embedding.detach().cpu().numpy()
        faiss.normalize_L2(emb)
        k = min(k, self.index.ntotal)  # Don't search for more than available
        _, indices = self.index.search(emb, k)
        return [self.metadata[i] for i in indices[0]]


# =========================================================
# 5. FULL VOICE PERSONA MODEL (Enhanced)
# =========================================================

class VoicePersonaModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.asr = WhisperASR()
        self.speaker_encoder = SpeakerEncoder()
        self.acoustic = AcousticFeatureExtractor()
        self.age_estimator = AgeEstimator()
        self.memory = VoiceMemory(dim=1024)  # XLS-R output size

    def preprocess(self, waveform, sr):
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
        return waveform.to(DEVICE)

    def enroll(self, waveform, sr, metadata):
        waveform = self.preprocess(waveform, sr)
        speaker_emb = self.speaker_encoder(waveform)
        
        # Auto-detect language and age if not provided
        if "language" not in metadata or metadata["language"] is None:
            _, detected_language = self.asr(waveform)
            metadata["language"] = detected_language
        
        if "age_group" not in metadata or metadata["age_group"] is None:
            acoustic_feat, _ = self.acoustic(waveform)
            age_group = self.age_estimator(acoustic_feat)
            metadata["age_group"] = age_group
        
        self.memory.add(speaker_emb, metadata)
        print(f"✅ Enrolled: {metadata}")

    @torch.no_grad()
    def forward(self, waveform, sr):
        waveform = self.preprocess(waveform, sr)

        # Transcription with language detection
        text, detected_language = self.asr(waveform)
        
        # Speaker embedding
        speaker_emb = self.speaker_encoder(waveform)
        
        # Acoustic features
        acoustic_feat, mel_spec = self.acoustic(waveform)
        
        # Age estimation
        estimated_age = self.age_estimator(acoustic_feat)

        # Search similar profiles
        matches = []
        if self.memory.index.ntotal > 0:
            matches = self.memory.search(speaker_emb)

        return {
            "transcription": text,
            "detected_language": detected_language,
            "estimated_age_group": estimated_age,
            "speaker_embedding": speaker_emb,
            "acoustic_features": acoustic_feat,
            "similar_profiles": matches
        }


# =========================================================
# 6. USAGE EXAMPLE
# =========================================================

if __name__ == "__main__":
    print("🚀 Initializing Voice Persona Model with Age & Language Detection...")
    print(f"📱 Device: {DEVICE}")
    
    # Check if sample file exists
    import os
    sample_file = "sample.wav"
    
    if not os.path.exists(sample_file):
        print(f"\n⚠️  Sample file '{sample_file}' not found!")
        print("📝 Creating a sample audio file for testing...")
        
        # Generate a simple test audio (sine wave)
        duration = 3  # seconds
        frequency = 440  # Hz (A4 note)
        t = torch.linspace(0, duration, int(SAMPLE_RATE * duration))
        waveform = torch.sin(2 * np.pi * frequency * t).unsqueeze(0)
        torchaudio.save(sample_file, waveform, SAMPLE_RATE)
        print(f"✅ Created test audio file: {sample_file}")
    
    # Load audio
    waveform, sr = torchaudio.load(sample_file)
    print(f"🎵 Loaded audio: {waveform.shape}, Sample Rate: {sr} Hz")

    # Initialize model
    model = VoicePersonaModel()

    # Enroll a voice profile (auto-detects language and age)
    print("\n📥 Enrolling voice profile...")
    model.enroll(
        waveform,
        sr,
        metadata={
            "region": "North India",
            "gender": "Male",
            "language": None,  # Will be auto-detected
            "age_group": None   # Will be auto-estimated
        }
    )

    # Analyze voice
    print("\n🔍 Analyzing voice...")
    output = model(waveform, sr)

    # Display results
    print("\n" + "="*60)
    print("📊 VOICE ANALYSIS RESULTS")
    print("="*60)
    print(f"🧠 TRANSCRIPTION: {output['transcription']}")
    print(f"🌍 DETECTED LANGUAGE: {output['detected_language']}")
    print(f"👤 ESTIMATED AGE GROUP: {output['estimated_age_group']}")
    print(f"🎧 ACOUSTIC FEATURES: {output['acoustic_features']}")
    print(f"🔍 SIMILAR PROFILES: {output['similar_profiles']}")
    print("="*60)
