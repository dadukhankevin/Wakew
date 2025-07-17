#!/usr/bin/env python3
"""
Multi-Wake Word Whisper Fine-tuning Script

Usage:
    python multi_trainer.py --wake_words "jeeves,jarvis,alexa" --audio_dirs "audio_Jeeves,audio_Jarvis,audio_Alexa"
    
Or auto-discover:
    python multi_trainer.py --wake_words "jeeves,jarvis,alexa" --auto_discover
"""

import os
import argparse
import torch
import librosa
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import numpy as np

from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    TrainingArguments,
    Trainer
)
from datasets import Dataset, Audio

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper for multiple wake words")
    parser.add_argument("--wake_words", type=str, required=True,
                       help="Comma-separated wake words (e.g., 'jeeves,jarvis,alexa')")
    parser.add_argument("--audio_dirs", type=str,
                       help="Comma-separated audio directories (e.g., 'audio_Jeeves,audio_Jarvis')")
    parser.add_argument("--auto_discover", action="store_true",
                       help="Auto-discover audio directories based on wake words")
    parser.add_argument("--model_name", type=str, default="openai/whisper-small",
                       help="Whisper model to fine-tune")
    parser.add_argument("--output_dir", type=str, default="./whisper-multi-wake-word",
                       help="Output directory for fine-tuned model")
    parser.add_argument("--train_split", type=float, default=0.8,
                       help="Fraction of data to use for training")
    parser.add_argument("--max_steps", type=int, default=200,
                       help="Maximum training steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Training batch size")
    parser.add_argument("--eval_steps", type=int, default=50,
                       help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, default=100,
                       help="Save frequency")
    return parser.parse_args()

def load_audio_files(audio_dir: str):
    """Load all audio files from directory"""
    audio_path = Path(audio_dir)
    if not audio_path.exists():
        return []
    
    audio_files = []
    for ext in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
        audio_files.extend(audio_path.glob(f'*{ext}'))
        audio_files.extend(audio_path.glob(f'**/*{ext}'))
    
    return [str(f) for f in audio_files]

def create_multi_wake_word_dataset(wake_words: List[str], audio_dirs: List[str], train_split: float = 0.8):
    """Create train/eval datasets from multiple wake word directories"""
    all_audio_files = []
    all_texts = []
    
    for wake_word, audio_dir in zip(wake_words, audio_dirs):
        audio_files = load_audio_files(audio_dir)
        if len(audio_files) == 0:
            continue
        
        all_audio_files.extend(audio_files)
        all_texts.extend([f" {wake_word}"] * len(audio_files))
        print(f"Added {len(audio_files)} samples for '{wake_word}'")
    
    if len(all_audio_files) == 0:
        raise ValueError("No audio files found")
    
    dataset = Dataset.from_dict({"audio": all_audio_files, "text": all_texts})
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.train_test_split(train_size=train_split, seed=42)
    
    print(f"Dataset: {len(all_audio_files)} total, {len(dataset['train'])} train, {len(dataset['test'])} eval")
    return dataset

def prepare_dataset(batch, processor):
    """Process audio and text for training"""
    audio = batch["audio"]
    
    if isinstance(audio["array"], np.ndarray):
        audio_array = audio["array"]
    else:
        audio_array = np.array(audio["array"])
    
    input_features = processor(
        audio_array, 
        sampling_rate=audio["sampling_rate"], 
        return_tensors="pt"
    ).input_features[0]
    
    labels = processor.tokenizer(
        batch["text"], 
        return_tensors="pt"
    ).input_ids[0]
    
    return {
        "input_features": input_features,
        "labels": labels
    }

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

def compute_metrics(eval_pred, processor):
    """Compute evaluation metrics"""
    pred_ids = eval_pred.predictions
    label_ids = eval_pred.label_ids

    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    
    import numpy as np
    if not isinstance(pred_ids, np.ndarray):
        pred_ids = np.array(pred_ids)
    if not isinstance(label_ids, np.ndarray):
        label_ids = np.array(label_ids)

    if len(pred_ids.shape) > 2:
        pred_ids = pred_ids.argmax(axis=-1)

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    try:
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        correct = sum(p.strip().lower() == l.strip().lower() for p, l in zip(pred_str, label_str))
        accuracy = correct / len(pred_str)
        
        return {"accuracy": accuracy}
    except Exception as e:
        print(f"Error in compute_metrics: {e}")
        return {"accuracy": 0.0}

def main():
    args = parse_args()
    
    # Parse wake words
    wake_words = [w.strip() for w in args.wake_words.split(',')]
    
    # Determine audio directories
    if args.auto_discover:
        audio_dirs = [f"audio_{word.title()}" for word in wake_words]
    elif args.audio_dirs:
        audio_dirs = [d.strip() for d in args.audio_dirs.split(',')]
    else:
        raise ValueError("Must specify either --audio_dirs or --auto_discover")
    
    if len(wake_words) != len(audio_dirs):
        raise ValueError(f"Wake words ({len(wake_words)}) must match directories ({len(audio_dirs)})")
    
    print(f"Training: {', '.join(wake_words)}")
    
    # Check device
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load model and processor
    print(f"\nLoading model and processor: {args.model_name}")
    processor = WhisperProcessor.from_pretrained(args.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    
    # Create multi-wake word dataset
    print("\nPreparing multi-wake word dataset...")
    dataset = create_multi_wake_word_dataset(wake_words, audio_dirs, args.train_split)
    
    # Process dataset
    print("Processing audio and text...")
    dataset = dataset.map(
        lambda batch: prepare_dataset(batch, processor),
        remove_columns=dataset["train"].column_names
    )
    
    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        warmup_steps=50,
        max_steps=args.max_steps,
        gradient_checkpointing=False,
        fp16=False,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=3,
        report_to=None,
        dataloader_pin_memory=False if device == "mps" else True,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, processor),
    )
    
    # Start training
    print(f"\nStarting training for {args.max_steps} steps...")
    trainer.train()
    
    # Save final model
    print(f"\nSaving final model to {args.output_dir}")
    trainer.save_model()
    processor.save_pretrained(args.output_dir)
    
    # Save wake words configuration
    wake_words_file = os.path.join(args.output_dir, "wake_words.txt")
    with open(wake_words_file, 'w') as f:
        for word in wake_words:
            f.write(f"{word}\n")
    print(f"Wake words saved to: {wake_words_file}")
    
    print("=== Training Complete! ===")
    print(f"Model saved to: {args.output_dir}")
    print(f"Supports wake words: {', '.join(wake_words)}")
    
    # Test inference on first sample of each wake word
    print(f"\nTesting model on samples...")
    test_data = dataset["test"]
    
    for wake_word in wake_words:
        # Find first sample of this wake word
        for i, sample in enumerate(test_data):
            if processor.tokenizer.decode(sample["labels"], skip_special_tokens=True).strip().lower() == wake_word.lower():
                input_features = torch.tensor(sample["input_features"]).unsqueeze(0)
                
                if device != "cpu":
                    model = model.to(device)
                    input_features = input_features.to(device)
                
                with torch.no_grad():
                    predicted_ids = model.generate(input_features, max_length=10)
                    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                
                print(f"  '{wake_word}' → '{transcription}'")
                break

if __name__ == "__main__":
    main() 