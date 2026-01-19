# 🎙️ Voice Persona Model with Age & Language Detection

An advanced voice analysis system that uses deep learning to extract speaker characteristics, detect language, estimate age, and build voice profiles.

## ✨ Features

- **🎧 Speech-to-Text**: Whisper-based transcription
- **🌍 Language Detection**: Automatic language identification from speech
- **👤 Age Estimation**: Heuristic-based age group classification
- **🔊 Speaker Embeddings**: Wav2Vec2-based speaker identification
- **🎵 Acoustic Analysis**: Pitch, energy, zero-crossing rate, spectral features
- **💾 Voice Memory**: FAISS-based similarity search for speaker profiles

## 📋 Requirements

- Python 3.8+
- CUDA (optional, for GPU acceleration)

## 🚀 Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For GPU support, install PyTorch with CUDA:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. For CPU-only (Mac/Linux):
```bash
pip install faiss-cpu
```

## 💻 Usage

### Basic Example

```python
import torch
import torchaudio
from voice_persona_model import VoicePersonaModel

# Initialize model
model = VoicePersonaModel()

# Load audio
waveform, sr = torchaudio.load("sample.wav")

# Enroll a voice profile
model.enroll(
    waveform,
    sr,
    metadata={
        "region": "North India",
        "gender": "Male",
        "language": None,  # Auto-detected
        "age_group": None   # Auto-estimated
    }
)

# Analyze voice
output = model(waveform, sr)

print("Transcription:", output['transcription'])
print("Language:", output['detected_language'])
print("Age Group:", output['estimated_age_group'])
print("Similar Profiles:", output['similar_profiles'])
```

### Run the Demo

```bash
python voice_persona_model.py
```

## 🧠 Model Components

### 1. WhisperASR
- **Model**: OpenAI Whisper Large V3
- **Function**: Speech-to-text transcription + language detection
- **Output**: Transcribed text and detected language
- **Important**: Converts batched tensors [1, T] to 1D numpy arrays [T] for proper Whisper processing

### 2. SpeakerEncoder
- **Model**: Wav2Vec2 XLS-R
- **Function**: Generate speaker embeddings
- **Output**: 1024-dim normalized embedding vector

### 3. AcousticFeatureExtractor
- **Function**: Extract acoustic features
- **Features**:
  - Pitch (fundamental frequency)
  - Energy (signal power)
  - Zero-crossing rate
  - Spectral centroid

### 4. AgeEstimator
- **Method**: Enhanced heuristic rules based on acoustic features
- **Age Groups** (8 categories):
  - Infant (0-3)
  - Child (4-12)
  - Teenager (13-17)
  - Young Adult (18-25)
  - Adult (26-40)
  - Middle-Aged (41-55)
  - Senior (56-70)
  - Elderly (71+)

### 5. VoiceMemory
- **Backend**: FAISS (Facebook AI Similarity Search)
- **Method**: Cosine similarity
- **Function**: Store and retrieve similar voice profiles

## 📊 Output Format

```python
{
    "transcription": "Hello, how are you?",
    "detected_language": "English",
    "estimated_age_group": "Adult (26-40)",
    "speaker_embedding": tensor([...]),  # 1024-dim vector
    "acoustic_features": tensor([...]),   # [pitch, energy, zcr, spectral_centroid]
    "similar_profiles": [                 # Top-K similar profiles
        {"region": "North India", "gender": "Male", ...}
    ]
}
```

## 🎯 Age Detection Algorithm

The age estimator uses enhanced acoustic features to classify speakers into 8 detailed age groups:

- **Pitch**: Higher pitch → Younger speaker
- **Spectral Centroid**: Higher frequencies → Younger speaker
- **Energy**: Voice strength and clarity
- **Zero-Crossing Rate**: Voice texture and quality

### Enhanced Heuristic Rules:
- **Pitch > 300 Hz + High Spectral Centroid** → Infant (0-3)
- **Pitch > 250 Hz + High Spectral Features** → Child (4-12)
- **Pitch > 200 Hz + High Energy** → Teenager (13-17)
- **Pitch > 160 Hz + Clear Voice** → Young Adult (18-25)
- **Pitch > 130 Hz + Stable Features** → Adult (26-40)
- **Pitch > 110 Hz + Moderate Features** → Middle-Aged (41-55)
- **Pitch > 90 Hz + Lower Energy** → Senior (56-70)
- **Lower Pitch + Reduced Energy** → Elderly (71+)

## 🌐 Supported Languages

Whisper supports 90+ languages including:
- English, Hindi, Spanish, French, German, Italian
- Portuguese, Russian, Japanese, Chinese, Arabic, Korean
- Regional Indian languages: Punjabi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Urdu

## ⚙️ Configuration

Edit the global config in `voice_persona_model.py`:

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
```

## 🔧 Advanced Usage

### Custom Enrollment

```python
# Manual metadata
model.enroll(
    waveform, sr,
    metadata={
        "name": "John Doe",
        "region": "Mumbai",
        "gender": "Male",
        "language": "Hindi",
        "age_group": "Adult (26-40)",
        "occupation": "Teacher"
    }
)
```

### Similarity Search

```python
# Find similar speakers
output = model(waveform, sr)
similar = output['similar_profiles']

for profile in similar:
    print(f"Match: {profile}")
```

## 🐛 Troubleshooting

### IndexError: index out of bounds in predicted_ids
**Cause**: Manually accessing token positions assumes fixed Whisper output structure
**Solution**: Use `processor.tokenizer.detect_language()` official API instead

### ValueError: operands could not be broadcast together
**Cause**: Whisper expects 1D audio arrays, not batched tensors [1, T]
**Solution**: The model automatically converts tensors to 1D numpy arrays

### CUDA Out of Memory
```python
# Use smaller Whisper model
self.asr = WhisperASR("openai/whisper-medium")
```

### Slow Processing
- Use GPU if available
- Use smaller models (whisper-base, whisper-small)
- Process shorter audio segments

### Import Errors
```bash
pip install --upgrade transformers torch torchaudio
```

## 📝 Notes

- First run will download models (~3-5 GB)
- Age estimation is heuristic-based (not ML-trained)
- For production use, train a dedicated age classifier
- Language detection works best with clear speech (>2 seconds)

## 🚀 Future Enhancements

- [ ] Train ML-based age classifier
- [ ] Emotion detection
- [ ] Gender classification
- [ ] Accent detection
- [ ] Real-time streaming support
- [ ] Multi-speaker diarization

## 🎯 Interview Tips

### Common Question: "Why was Whisper failing with broadcast errors?"

**Answer**: 
"Whisper expects 1D raw audio arrays for processing. I was initially passing batched tensors with shape [1, T], which caused padding broadcast failures during feature extraction. The solution was to convert the batched tensor to a 1D numpy array using `.squeeze().cpu().numpy()` before passing it to the WhisperProcessor. This resolved both transcription and language detection issues."

### Common Question: "How do you detect language in Whisper?"

**Answer**:
"I use Whisper's official built-in API: `processor.tokenizer.detect_language(input_features)`. Manually accessing token positions like `predicted_ids[0][1]` is unreliable because Whisper's output structure varies based on the audio and generation parameters. The official API is stable, production-ready, and matches OpenAI's reference implementation."

### Technical Highlights:
- **Whisper Input**: Requires 1D float arrays (T,) not (1, T)
- **Language Detection**: Uses Whisper's official `processor.tokenizer.detect_language()` API
- **Speaker Embeddings**: Wav2Vec2FeatureExtractor (not Processor) to avoid tokenizer errors
- **Age Estimation**: Heuristic-based using pitch, energy, and spectral features
- **No Token Index Assumptions**: Avoids brittle token position-based detection

## 📄 License

MIT License

## 🙏 Acknowledgments

- OpenAI Whisper
- Facebook Wav2Vec2
- FAISS
- Hugging Face Transformers
