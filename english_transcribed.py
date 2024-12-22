import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

def load_model_and_processor():
    model_name = "facebook/wav2vec2-large-960h"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    return model, processor

def resample_audio(audio_path, target_sample_rate=16000):
    waveform, original_sample_rate = torchaudio.load(audio_path)
    if original_sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    return waveform.squeeze(), target_sample_rate

def transcribe_audio(file_path: str, model, processor, device):
    # Resampling audio to 16kHz
    audio_input, sample_rate = resample_audio(file_path, target_sample_rate=16000)
    # Normalizing waveform
    audio_input = audio_input / torch.max(torch.abs(audio_input))
    # Preprocessing the audio input
    inputs = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt", padding=True).input_values.to(device)
    # Performing  transcription
    with torch.no_grad():
        logits = model(inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    # Decoding the transcription
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

if __name__ == "__main__":
    # Setting up the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Loading model and processor
    model, processor = load_model_and_processor()
    model = model.to(device)

    # File path for the audio
    audio_file = "~/english.wav"

    # Transcribing the audio
    transcription = transcribe_audio(audio_file, model, processor, device)
    print("Transcribed English Output:", transcription)
