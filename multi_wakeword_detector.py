#!/usr/bin/env python3

import torch
import numpy as np
import sounddevice as sd
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import webrtcvad
import os

class MultiWakeWordDetector:
    def __init__(self, model_path, wake_words=None, callback=None, sample_rate=16000, verbose=False):
        """
        Initialize multi-wake word detector.
        
        Args:
            model_path: Path to fine-tuned Whisper model
            wake_words: List of wake words to detect or None to auto-load from model
            callback: Function to call when wake word is detected (receives wake_word, transcription)
            sample_rate: Audio sample rate (default 16000)
            verbose: Print debug information
        """
        self.model_path = model_path
        self.callback = callback
        self.sample_rate = sample_rate
        self.verbose = verbose
        
        # Load wake words
        if wake_words is None:
            self.wake_words = self._load_wake_words_from_model()
        else:
            self.wake_words = [w.lower() for w in wake_words]
        
        if verbose:
            print(f"Loading multi-wake word detector for: {', '.join(self.wake_words)}")
        
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        
        # Auto-detect device
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        self.model = self.model.to(self.device)
        self.vad = webrtcvad.Vad(3)
        
        # Audio processing state
        self.chunk_size = int(sample_rate * 0.5)
        self.buffer_size = int(sample_rate * 2.0)
        self.speech_buffer = []
        self.silence_chunks = 0
        self.max_silence_chunks = 4  # Increased from 2 to keep recording longer
        self.is_recording = False
        
        if verbose:
            print(f"Ready! Listening for wake words on {self.device}")
            print(f"Wake words: {', '.join(self.wake_words)}")
    
    def _load_wake_words_from_model(self):
        """Load wake words from model directory"""
        wake_words_file = os.path.join(self.model_path, "wake_words.txt")
        if os.path.exists(wake_words_file):
            with open(wake_words_file, 'r') as f:
                wake_words = [line.strip().lower() for line in f.readlines() if line.strip()]
            if self.verbose:
                print(f"Loaded wake words from {wake_words_file}: {wake_words}")
            return wake_words
        else:
            if self.verbose:
                print(f"No wake_words.txt found in {self.model_path}, you must specify wake_words manually")
            return []
    
    def is_speech(self, audio_chunk):
        """Check if audio contains speech using energy + WebRTC VAD"""
        energy = np.mean(np.abs(audio_chunk))
        if energy < 0.01:
            return False
        
        frame_size = int(self.sample_rate * 0.03)
        speech_frames = 0
        total_frames = 0
        
        for i in range(0, len(audio_chunk), frame_size):
            frame = audio_chunk[i:i + frame_size]
            if len(frame) == frame_size:
                total_frames += 1
                frame_pcm = (frame * 32767).astype(np.int16).tobytes()
                if self.vad.is_speech(frame_pcm, self.sample_rate):
                    speech_frames += 1
        
        return total_frames > 0 and speech_frames / total_frames >= 0.5
    
    def process_audio(self, audio_data):
        """Process audio with Whisper model"""
        input_features = self.processor(
            audio_data, 
            sampling_rate=self.sample_rate, 
            return_tensors="pt"
        ).input_features.to(self.device)
        
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features, max_length=50)
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        return transcription.strip()
    
    def detect_wake_words(self, transcription):
        """Check if any wake words are in the transcription"""
        transcription_lower = transcription.lower()
        detected_words = []
        
        for wake_word in self.wake_words:
            if wake_word in transcription_lower:
                detected_words.append(wake_word)
        
        return detected_words
    
    def on_wake_words_detected(self, detected_words, transcription):
        """Handle wake word detection"""
        if self.verbose:
            if len(detected_words) == 1:
                print(f"🚨 WAKE WORD '{detected_words[0].upper()}' DETECTED!")
            else:
                print(f"🚨 WAKE WORDS {[w.upper() for w in detected_words]} DETECTED!")
        
        if self.callback:
            for wake_word in detected_words:
                self.callback(wake_word, transcription)
    
    def audio_callback(self, indata, frames, time, status):
        """Audio stream callback"""
        audio_chunk = indata[:, 0]
        
        if self.is_speech(audio_chunk):
            if not self.is_recording:
                self.is_recording = True
                self.speech_buffer = []
            
            self.speech_buffer.extend(audio_chunk)
            self.silence_chunks = 0
        else:
            if self.is_recording:
                # Add silence chunk to buffer to avoid cutting off words
                self.speech_buffer.extend(audio_chunk)
                self.silence_chunks += 1
                if self.silence_chunks >= self.max_silence_chunks:
                    if len(self.speech_buffer) > 0:
                        speech_audio = np.array(self.speech_buffer)
                        if len(speech_audio) >= self.sample_rate * 0.5:
                            transcription = self.process_audio(speech_audio)
                            
                            if self.verbose:
                                print(f"Heard: '{transcription}'")
                            
                            detected_words = self.detect_wake_words(transcription)
                            if detected_words:
                                self.on_wake_words_detected(detected_words, transcription)
                    
                    self.is_recording = False
                    self.speech_buffer = []
    
    def start_listening(self):
        """Start listening for wake words"""
        if len(self.wake_words) == 0:
            raise ValueError("No wake words specified. Either pass wake_words or ensure wake_words.txt exists in model directory.")
        
        if self.verbose:
            print("Listening...")
        
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.chunk_size,
            callback=self.audio_callback,
            dtype=np.float32
        ):
            try:
                while True:
                    sd.sleep(100)
            except KeyboardInterrupt:
                if self.verbose:
                    print("\nStopped listening")
    
    def start_listening_with_input(self):
        """Start listening with press Enter to stop"""
        if len(self.wake_words) == 0:
            raise ValueError("No wake words specified. Either pass wake_words or ensure wake_words.txt exists in model directory.")
        
        if self.verbose:
            print("Listening... (Press Enter to stop)")
        
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.chunk_size,
            callback=self.audio_callback,
            dtype=np.float32
        ):
            input()

# Example usage
def example_callback(wake_word, transcription):
    print(f"🎉 Wake word '{wake_word}' detected in: '{transcription}'")
    
    # Custom logic based on which wake word was detected
    if wake_word == "jeeves":
        print("🤵 Jeeves at your service!")
    elif wake_word == "jarvis":
        print("🤖 JARVIS online!")
    elif wake_word == "alexa":
        print("🔵 Alexa activated!")
    elif wake_word == "computer":
        print("💻 Computer ready!")

def main():
    # Example: Multi-wake word detector
    detector = MultiWakeWordDetector(
        model_path="./whisper-multi-wake-word",
        # wake_words=["jeeves", "jarvis", "alexa", "computer"],  # Optional - auto-loads from model
        callback=example_callback,
        verbose=True
    )
    
    detector.start_listening_with_input()

if __name__ == "__main__":
    main() 