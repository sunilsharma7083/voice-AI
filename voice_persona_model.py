import torch
import torchaudio
import faiss
import numpy as np
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Wav2Vec2Processor,
    Wav2Vec2Model,
    Wav2Vec2FeatureExtractor,
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
    pipeline
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
        inputs = self.processor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        ).to(DEVICE)

        # Generate with return_dict_in_generate for language detection
        outputs = self.model.generate(
            inputs.input_features,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        ids = outputs.sequences
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0]
        
        # Detect language from the generated tokens
        detected_language = self.detect_language(inputs.input_features)
        
        return text, detected_language

    @torch.no_grad()
    def detect_language(self, input_features):
        """Detect language using Whisper's built-in language detection"""
        try:
            # Get language tokens from Whisper
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=None, task="transcribe")
            outputs = self.model.generate(
                input_features,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            # Extract language from tokens
            # Whisper encodes language in special tokens
            language_token = outputs.sequences[0][1].item()
            language_map = {
                50259: "en", 50260: "zh", 50261: "de", 50262: "es",
                50263: "ru", 50264: "ko", 50265: "fr", 50266: "ja",
                50267: "pt", 50268: "tr", 50269: "pl", 50270: "ca",
                50271: "nl", 50272: "ar", 50273: "sv", 50274: "it",
                50275: "id", 50276: "hi", 50277: "fi", 50278: "vi",
                50279: "he", 50280: "uk", 50281: "el", 50282: "ms",
                50283: "cs", 50284: "ro", 50285: "da", 50286: "hu",
                50287: "ta", 50288: "no", 50289: "th", 50290: "ur",
                50291: "hr", 50292: "bg", 50293: "lt", 50294: "la",
                50295: "mi", 50296: "ml", 50297: "cy", 50298: "sk",
                50299: "te", 50300: "fa", 50301: "lv", 50302: "bn",
                50303: "sr", 50304: "az", 50305: "sl", 50306: "kn",
                50307: "et", 50308: "mk", 50309: "br", 50310: "eu",
                50311: "is", 50312: "hy", 50313: "ne", 50314: "mn",
                50315: "bs", 50316: "kk", 50317: "sq", 50318: "sw",
                50319: "gl", 50320: "mr", 50321: "pa", 50322: "si",
                50323: "km", 50324: "sn", 50325: "yo", 50326: "so",
                50327: "af", 50328: "oc", 50329: "ka", 50330: "be",
                50331: "tg", 50332: "sd", 50333: "gu", 50334: "am",
                50335: "yi", 50336: "lo", 50337: "uz", 50338: "fo",
                50339: "ht", 50340: "ps", 50341: "tk", 50342: "nn",
                50343: "mt", 50344: "sa", 50345: "lb", 50346: "my",
                50347: "bo", 50348: "tl", 50349: "mg", 50350: "as",
                50351: "tt", 50352: "haw", 50353: "ln", 50354: "ha",
                50355: "ba", 50356: "jw", 50357: "su"
            }
            
            detected_lang = language_map.get(language_token, "unknown")
            
            # Map to full language names
            lang_names = {
                "en": "English", "hi": "Hindi", "es": "Spanish",
                "fr": "French", "de": "German", "it": "Italian",
                "pt": "Portuguese", "ru": "Russian", "ja": "Japanese",
                "zh": "Chinese", "ar": "Arabic", "ko": "Korean",
                "pa": "Punjabi", "bn": "Bengali", "ta": "Tamil",
                "te": "Telugu", "mr": "Marathi", "gu": "Gujarati",
                "kn": "Kannada", "ml": "Malayalam", "ur": "Urdu"
            }
            
            return lang_names.get(detected_lang, detected_lang.upper())
        except:
            return "Unknown"


# =========================================================
# 2. BRAIN-A: Speaker Embeddings (Wav2Vec2 / XLS-R)
# =========================================================

class SpeakerEncoder(torch.nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-large-xlsr-53"):
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
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
            torch.nn.Linear(4, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(32, 4)  # 4 age groups
        ).to(DEVICE)
        
        self.age_groups = ["Child (0-12)", "Young Adult (13-30)", "Adult (31-50)", "Senior (51+)"]
    
    @torch.no_grad()
    def forward(self, acoustic_features):
        """Estimate age group from acoustic features"""
        # Rule-based age estimation (heuristic approach)
        pitch_mean = acoustic_features[0].item()
        energy_mean = acoustic_features[1].item()
        spectral_centroid = acoustic_features[3].item()
        
        # Heuristic rules for age estimation
        if pitch_mean > 250:  # Hz
            age_group = "Child (0-12)"
        elif pitch_mean > 180 and spectral_centroid > 2000:
            age_group = "Young Adult (13-30)"
        elif pitch_mean > 120:
            age_group = "Adult (31-50)"
        else:
            age_group = "Senior (51+)"
        
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
