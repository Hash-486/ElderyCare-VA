import torch
import whisper
import yaml
from pathlib import Path

def load_stt_model(config_path: str = "config/pipeline_config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    model_size = config["stt"]["model"]
    device = config["stt"]["device"]
    model = whisper.load_model(model_size, device=device)
    return model

def transcribe_audio(audio_path: str, model) -> str:
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    result = model.transcribe(audio, language="en", fp16=False)
    return result["text"].strip()


if __name__ == "__main__":
    model = load_stt_model()
    print("✅ STT model loaded successfully!")
