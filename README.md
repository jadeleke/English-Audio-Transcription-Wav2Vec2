# Wav2Vec2 English Audio Transcriber

This repository demonstrates how to use Facebook's "wav2vec2-large-960h" pre-trained model for transcribing English audio. The script is built using PyTorch, Torchaudio, and the Hugging Face Transformers library.

## Features
- **Resampling Audio**: Converts input audio to a target sample rate of 16,000 Hz.
- **Waveform Normalization**: Ensures consistent input scaling for better transcription accuracy.
- **End-to-End Transcription**: Processes and transcribes English audio efficiently.
- **GPU Support**: Automatically leverages GPU for faster execution if available.

## Requirements
To use the script, ensure you have the following installed:

- Python 3.7+
- PyTorch
- Torchaudio
- Transformers (by Hugging Face)

Install the dependencies with:
```bash
pip install torch torchaudio transformers
```

## Code Explanation

### Loading the Model and Processor
The Wav2Vec2 model and processor are loaded from Hugging Face's Transformers library.
```python
def load_model_and_processor():
    model_name = "facebook/wav2vec2-large-960h"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    return model, processor
```

### Resampling Audio
Audio is resampled to the target sample rate of 16,000 Hz to meet the model's requirements.
```python
def resample_audio(audio_path, target_sample_rate=16000):
    waveform, original_sample_rate = torchaudio.load(audio_path)
    if original_sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    return waveform.squeeze(), target_sample_rate
```

### Transcribing Audio
The transcription process includes:
1. Resampling and normalizing the audio waveform.
2. Preprocessing the waveform for the model.
3. Generating predictions and decoding them into text.

```python
def transcribe_audio(file_path: str, model, processor, device):
    audio_input, sample_rate = resample_audio(file_path, target_sample_rate=16000)
    audio_input = audio_input / torch.max(torch.abs(audio_input))
    inputs = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt", padding=True).input_values.to(device)
    with torch.no_grad():
        logits = model(inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription
```

### Example Usage
The script is designed to be run directly, with the main function handling device setup, model loading, and audio transcription.
```python
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_model_and_processor()
    model = model.to(device)
    audio_file = "/Users/josephadeleke/Documents/dataset/english.wav"
    transcription = transcribe_audio(audio_file, model, processor, device)
    print("Transcribed English Output:", transcription)
```

Replace `audio_file` with the path to your audio file.

## Running the Script
1. Save the script as `transcribe_english.py`.
2. Place your audio file in the desired location.
3. Run the script with:
```bash
python transcribe_english.py
```
The transcribed output will be displayed in the console.

## Contribution
Contributions are welcome! Feel free to submit issues or pull requests to enhance functionality.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

