# Wakew: Wakeword Detection via Finetuned Whisper Models using Synthetic data.

No, I don't know how it's pronounced.

Real-time wake word detection using fine-tuned Whisper models. Detects wake words anywhere in speech, supports multiple wake words, uses synthetic training data, and includes voice assistant features.

## Features

- Single & multi-wake word support
- Detects wake words anywhere in speech 
- Real-time processing with voice activity detection
- GPU acceleration (CUDA/MPS/CPU)
- **Voice assistant integration** with interruption detection
- **Simple transcription** - transcribe next speech input
- **Threading support** for non-blocking callbacks
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

### Basic Wake Word Detection

```python
MultiWakeWordDetector(model_path, wake_words=None, callback=None, 
                     interruption_callback=None, verbose=False)
```

**Parameters:**
- `model_path`: Path to trained model
- `wake_words`: List of wake words (auto-loads from model if None)  
- `callback`: Function called with (wake_word, transcription)
- `interruption_callback`: Function called when speech interrupts assistant response
- `verbose`: Print debug info

**Methods:**
- `start_listening()` - Start continuous wake word detection
- `start_listening_with_input()` - Start detection, stop with Enter key
- `transcribe_next_speech(timeout=10.0)` - Wait for speech and transcribe it
- `set_assistant_speaking(speaking)` - Set assistant speaking state for interruption detection

### Voice Assistant Integration

```python
def wake_word_callback(wake_word, transcription):
    # Generate response
    response = generate_response(transcription)
    
    # Tell detector assistant is speaking
    detector.set_assistant_speaking(True)
    
    # Play TTS response
    play_tts(response)
    
    # Done speaking
    detector.set_assistant_speaking(False)

def interruption_callback():
    # User interrupted - stop TTS
    stop_tts()
    detector.set_assistant_speaking(False)

detector = MultiWakeWordDetector(
    model_path="./whisper-multi-wake-word",
    callback=wake_word_callback,
    interruption_callback=interruption_callback
)
```

### Simple Transcription

```python
detector = MultiWakeWordDetector("./whisper-multi-wake-word")

# Wait for user speech and transcribe
user_input = detector.transcribe_next_speech(timeout=15.0)
if user_input:
    print(f"User said: {user_input}")
```

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
