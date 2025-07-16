#!/usr/bin/env python3

import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os
from pathlib import Path

# Model paths
ORIGINAL_MODEL = "openai/whisper-small"
MULTI_MODEL = "./whisper-multi-wake-word"

def load_models():
    """Load both original and multi-wake word models"""
    print("Loading models...")
    
    # Load original model
    print("📥 Loading original Whisper model...")
    original_processor = WhisperProcessor.from_pretrained(ORIGINAL_MODEL)
    original_model = WhisperForConditionalGeneration.from_pretrained(ORIGINAL_MODEL)
    
    # Load multi-wake word model
    print("📥 Loading multi-wake word model...")
    multi_processor = WhisperProcessor.from_pretrained(MULTI_MODEL)
    multi_model = WhisperForConditionalGeneration.from_pretrained(MULTI_MODEL)
    
    # Device setup
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA GPU")
    else:
        device = "cpu"
        print("Using CPU")
    
    original_model = original_model.to(device)
    multi_model = multi_model.to(device)
    
    return (original_processor, original_model), (multi_processor, multi_model), device

def load_wake_words():
    """Load wake words from multi-model directory"""
    wake_words_file = os.path.join(MULTI_MODEL, "wake_words.txt")
    if os.path.exists(wake_words_file):
        with open(wake_words_file, 'r') as f:
            wake_words = [line.strip().lower() for line in f.readlines() if line.strip()]
        return wake_words
    else:
        # Default wake words if no file exists
        return ["jeeves", "jarvis", "alexa", "computer"]

def find_audio_directories(wake_words):
    """Find audio directories for each wake word"""
    audio_dirs = {}
    for word in wake_words:
        # Try different directory naming conventions
        possible_dirs = [
            f"audio_{word}",
            f"audio_{word.title()}",
            f"audio_{word.upper()}",
            f"audio_{word.lower()}"
        ]
        
        for dir_name in possible_dirs:
            if os.path.exists(dir_name):
                audio_dirs[word] = dir_name
                break
        
        if word not in audio_dirs:
            print(f"⚠️  No audio directory found for '{word}'")
    
    return audio_dirs

def load_audio_files(audio_dir):
    """Load all audio files from directory"""
    audio_path = Path(audio_dir)
    audio_files = []
    extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    
    for ext in extensions:
        audio_files.extend(list(audio_path.glob(f'*{ext}')))
        audio_files.extend(list(audio_path.glob(f'**/*{ext}')))
    
    return [str(f) for f in audio_files]

def test_model(audio_path, processor, model, device):
    """Test a single model on an audio file"""
    # Load and process audio
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Process with Whisper processor
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    input_features = input_features.to(device)
    
    # Generate transcription
    with torch.no_grad():
        predicted_ids = model.generate(input_features, max_length=50)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    return transcription

def evaluate_wake_word_detection(transcription, expected_word):
    """Check if the expected wake word is detected in transcription"""
    return expected_word.lower() in transcription.lower()

def main():
    # Load models
    (original_processor, original_model), (multi_processor, multi_model), device = load_models()
    
    # Load wake words and find audio directories
    wake_words = load_wake_words()
    audio_dirs = find_audio_directories(wake_words)
    
    print(f"\n🎯 Testing wake words: {', '.join(wake_words)}")
    print(f"📁 Audio directories: {audio_dirs}")
    
    # Test each wake word
    results = {}
    
    for wake_word in wake_words:
        if wake_word not in audio_dirs:
            continue
        
        print(f"\n=== Testing '{wake_word.upper()}' ===")
        
        # Load audio files for this wake word
        audio_files = load_audio_files(audio_dirs[wake_word])
        if len(audio_files) == 0:
            print(f"No audio files found for {wake_word}")
            continue
        
        # Test both models
        original_correct = 0
        multi_correct = 0
        
        for i, audio_file in enumerate(audio_files):
            print(f"Testing {i+1}/{len(audio_files)}: {os.path.basename(audio_file)}")
            
            # Test original model
            original_transcription = test_model(audio_file, original_processor, original_model, device)
            original_detected = evaluate_wake_word_detection(original_transcription, wake_word)
            if original_detected:
                original_correct += 1
            
            # Test multi-wake word model
            multi_transcription = test_model(audio_file, multi_processor, multi_model, device)
            multi_detected = evaluate_wake_word_detection(multi_transcription, wake_word)
            if multi_detected:
                multi_correct += 1
            
            # Show comparison for this file
            status_original = "✅" if original_detected else "❌"
            status_multi = "✅" if multi_detected else "❌"
            
            print(f"  Original: {status_original} '{original_transcription}'")
            print(f"  Multi:    {status_multi} '{multi_transcription}'")
        
        # Calculate accuracy for this wake word
        original_accuracy = (original_correct / len(audio_files)) * 100
        multi_accuracy = (multi_correct / len(audio_files)) * 100
        improvement = multi_accuracy - original_accuracy
        
        results[wake_word] = {
            'total_files': len(audio_files),
            'original_correct': original_correct,
            'multi_correct': multi_correct,
            'original_accuracy': original_accuracy,
            'multi_accuracy': multi_accuracy,
            'improvement': improvement
        }
        
        print(f"\n📊 Results for '{wake_word}':")
        print(f"   Original Whisper: {original_correct}/{len(audio_files)} ({original_accuracy:.1f}%)")
        print(f"   Multi-Wake Model: {multi_correct}/{len(audio_files)} ({multi_accuracy:.1f}%)")
        print(f"   Improvement: {improvement:+.1f} percentage points")
    
    # Overall summary
    if results:
        print(f"\n" + "="*50)
        print("🏆 OVERALL PERFORMANCE SUMMARY")
        print("="*50)
        
        total_files = sum(r['total_files'] for r in results.values())
        total_original_correct = sum(r['original_correct'] for r in results.values())
        total_multi_correct = sum(r['multi_correct'] for r in results.values())
        
        overall_original_accuracy = (total_original_correct / total_files) * 100
        overall_multi_accuracy = (total_multi_correct / total_files) * 100
        overall_improvement = overall_multi_accuracy - overall_original_accuracy
        
        print(f"📈 Overall Accuracy:")
        print(f"   Original Whisper: {total_original_correct}/{total_files} ({overall_original_accuracy:.1f}%)")
        print(f"   Multi-Wake Model: {total_multi_correct}/{total_files} ({overall_multi_accuracy:.1f}%)")
        print(f"   Overall Improvement: {overall_improvement:+.1f} percentage points")
        
        print(f"\n📋 Per-Wake Word Results:")
        for wake_word, result in results.items():
            print(f"   {wake_word.title()}: {result['original_accuracy']:.1f}% → {result['multi_accuracy']:.1f}% ({result['improvement']:+.1f}pp)")
        
        # Best and worst performing wake words
        best_word = max(results.keys(), key=lambda w: results[w]['multi_accuracy'])
        worst_word = min(results.keys(), key=lambda w: results[w]['multi_accuracy'])
        
        print(f"\n🥇 Best performing: '{best_word}' ({results[best_word]['multi_accuracy']:.1f}%)")
        print(f"🔍 Needs improvement: '{worst_word}' ({results[worst_word]['multi_accuracy']:.1f}%)")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        for wake_word, result in results.items():
            if result['multi_accuracy'] < 90:
                print(f"   • Consider more training data for '{wake_word}' (current: {result['multi_accuracy']:.1f}%)")
        
        if overall_improvement > 0:
            print(f"✅ Multi-wake word training was successful!")
        else:
            print(f"⚠️  Multi-wake word training may need adjustment")

if __name__ == "__main__":
    main() 