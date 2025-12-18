import torch
import whisper
import yaml
from pathlib import Path
import os

# Create a YAML config file for Whisper model settings
config_data = {
    "stt": {
        "model": "base", # or "tiny", "small", "medium", "large"
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
}

config_file_path = "/content/whisper_config.yaml"
with open(config_file_path, "w") as f:
    yaml.dump(config_data, f)

class VoiceToTextPipeline:
    def __init__(self, config_path: str = config_file_path):
        self.config_path = config_path
        self.model = self.load_stt_model()

    def load_stt_model(self):
        """Load Whisper model based on config."""
        with open(self.config_path) as f:
            config = yaml.safe_load(f)
        model_size = config["stt"]["model"]
        device = config["stt"]["device"]
        print(f"Loading Whisper model: {model_size} on {device}...")
        model = whisper.load_model(model_size, device=device)
        print("✅ STT model loaded successfully!")
        return model

    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio file to text using Whisper.
        Supports .wav, .mp3, etc.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"Transcribing audio: {audio_path}")
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        result = self.model.transcribe(audio, language="en", fp16=False)
        text = result["text"].strip()
        print(f"✅ Transcription complete!")
        return text

    def detect_and_display_text(self, audio_path: str) -> str:
        """
        Complete pipeline: Load audio -> STT -> Display text
        """
        print(f"🎤 Processing audio file: {audio_path}")

        # Transcribe
        raw_text = self.transcribe_audio(audio_path)

        # Display the detected text
        print("\n" + "="*50)
        print("📝 DETECTED TEXT FROM AUDIO:")
        print("-" * 50)
        print(raw_text)
        print("="*50 + "\n")

        return raw_text

# Example usage and testing
if __name__ == "__main__":
    # Initialize the pipeline
    stt_pipeline = VoiceToTextPipeline()

    # Test with a sample audio file (you'll need to provide one)
    # For now, let's create a dummy test
    print("Testing STT pipeline...")

    # To test with actual audio, use:
    audio_file = "/content/harvard.wav"
    detected_text = stt_pipeline.detect_and_display_text(audio_file)

