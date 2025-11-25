# standard libraries
import warnings

warnings.filterwarnings("ignore")
from transformers import pipeline
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# -----------------------------------------------
# -----------------------------------------------
# SPEECH _ TO _ TEXT
# Components of speech
# pitch / Stress / Rhythm
# Audio-Recording --> ENCODE --> FEATURES --> DECODE --> AUDIO & TEXT
# text and Audio are sequential

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

# Lets try transcribing audio
# Use librispeech_asr instead (Parquet-backed, no script dependency)
dataset = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
)
# Use decode=False to avoid torchcodec dependency
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000, decode=False))
################################################
################################################
################################################
# Listen to a sample recording
import io
import soundfile as sf
from IPython.display import Audio as IPAudio, display

sample = dataset[18]
audio_bytes = sample["audio"]["bytes"]

# Decode bytes to waveform
with io.BytesIO(audio_bytes) as f:
    audio_array, sampling_rate = sf.read(f, dtype="float32")

print(f"Text: {sample['text']}")
display(IPAudio(audio_array, rate=sampling_rate))
################################################
################################################
################################################
# Preprocess audio for Whisper
input_preprocessed = processor(
    audio_array, sampling_rate=sampling_rate, return_tensors="pt"
)

# extract features from preprocessed audio
predicted_ids = model.generate(input_preprocessed.input_features, language="en")
# decode ids to natural language
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(f"Transcription: {transcription[0]}")
# -----------------------------------------------
# -----------------------------------------------
# -----------------------------------------------
# AUDIO GENERATION
# three components to generate audio
# Preprocessor: resampling and feature extraction
# Model: feature transformation
# Vocoder: a sepoerate generative model for audi waveforms
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch

# Load SpeechT5 components (voice cloning / TTS)
tts_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_vc")
tts_vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Obtain a deterministic synthetic speaker embedding (no dataset scripts)
speaker_embedding = torch.randn(1, 512)

# Text to synthesize
text_to_speak = "Hello there, this is a SpeechT5 demo without extra libraries."
inputs = tts_processor(text=text_to_speak, return_tensors="pt")

# Generate waveform directly using generate_speech helper
with torch.no_grad():
    speech_waveform = tts_model.generate_speech(
        inputs["input_ids"],
        speaker_embeddings=speaker_embedding,
        vocoder=tts_vocoder,
    )  # Tensor [samples]

print("Generated speech length (samples):", speech_waveform.shape[0])
display(IPAudio(speech_waveform.cpu().numpy(), rate=16000))
