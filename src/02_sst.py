# src/01_stt.py
import whisper
import yaml
import os
import torch
from pathlib import Path

class VoiceToTextPipeline:
    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        self.config_path = config_path
        self.model = self.load_stt_model()

    def load_stt_model(self):
        """Load Whisper model based on config."""
        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        model_size = config["stt"]["model"]
        device = config["stt"]["device"]

        print(f"🔊 Loading Whisper model: {model_size} on {device}...")
        model = whisper.load_model(model_size, device=device)
        print("✅ STT model loaded successfully!")
        return model

    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio file to text using Whisper.
        Supports .wav, .mp3, etc.
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"❌ Audio file not found: {audio_path}")

        print(f"🎤 Transcribing audio: {audio_path}")

        audio = whisper.load_audio(str(audio_path))
        audio = whisper.pad_or_trim(audio)

        use_fp16 = torch.cuda.is_available()
        result = self.model.transcribe(
            audio,
            language="en",
            fp16=use_fp16
        )

        text = result["text"].strip()
        print("✅ Transcription complete!")
        return text

    def detect_and_display_text(self, audio_path: str) -> str:
        """
        Complete pipeline: Load audio -> STT -> Display text
        """
        print(f"\n🎧 Processing audio file: {audio_path}")

        raw_text = self.transcribe_audio(audio_path)

        print("\n" + "=" * 60)
        print("📝 DETECTED TEXT FROM AUDIO")
        print("=" * 60)
        print(raw_text)
        print("=" * 60 + "\n")

        return raw_text


# ===========================
# RUN DIRECTLY
# ===========================
if __name__ == "__main__":
    stt_pipeline = VoiceToTextPipeline()

    # 🔴 YOUR ACTUAL AUDIO FILE PATH
    audio_file = r"C:\Users\kvenk\Documents\Elderly_Voice\src\harvard.wav"

    detected_text = stt_pipeline.detect_and_display_text(audio_file)
