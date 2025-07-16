# Wakew Wakeword Detection via finetuned Whisper models using Synthetic data.

Real-time wake word detection using fine-tuned Whisper models. Detects wake words anywhere in speech, supports multiple wake words, and uses synthetic training data.

## Features

- Single & multi-wake word support
- Detects wake words anywhere in speech 
- Real-time processing with voice activity detection
- GPU acceleration (CUDA/MPS/CPU)
- Simple API

## Quick Start

### Installation

```bash
pip install torch transformers datasets sounddevice webrtcvad librosa numpy tqdm
```

### Single Wake Word

```python
# 1. Generate training data
WORD = "jarvis"  # In word_speech_generator.py
python word_speech_generator.py

# 2. Train model  
python multi_trainer.py --wake_words "jarvis" --auto_discover --max_steps 50

# 3. Use detector
from multi_wakeword_detector import MultiWakeWordDetector

def callback(wake_word, transcription):
    print(f"{wake_word} detected: {transcription}")

detector = MultiWakeWordDetector("./whisper-multi-wake-word", callback=callback)
detector.start_listening()
```

### Multiple Wake Words

```python
# 1. Generate training data
WORDS = ["jarvis", "alexa", "computer"]  # In word_speech_generator.py
python word_speech_generator.py

# 2. Train model
python multi_trainer.py --wake_words "jarvis,alexa,computer" --auto_discover --max_steps 200

# 3. Use detector (same as above)
```

## API

```python
MultiWakeWordDetector(model_path, wake_words=None, callback=None, verbose=False)
```

- `model_path`: Path to trained model
- `wake_words`: List of wake words (auto-loads from model if None)  
- `callback`: Function called with (wake_word, transcription)
- `verbose`: Print debug info

Methods: `start_listening()`, `start_listening_with_input()`

## Training

Set `WORDS = ["your_words"]` in `word_speech_generator.py`, then:

```bash
python word_speech_generator.py
python multi_trainer.py --wake_words "word1,word2" --auto_discover --max_steps 200
python compare_multi_models.py  # Test performance
```

## Demo

```bash
python demo.py
```

## Files

- `word_speech_generator.py` - Generate training audio
- `multi_trainer.py` - Train wake word models  
- `multi_wakeword_detector.py` - Real-time detection
- `compare_multi_models.py` - Test model performance
- `demo.py` - Interactive demo 
