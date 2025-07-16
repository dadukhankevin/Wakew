#!/usr/bin/env python3

import os
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# CHANGE THIS TO GENERATE VARIATIONS FOR MULTIPLE WAKE WORDS
WORDS = ["Jeeves", "Jarvis", "Alexa", "Computer"]

class MultiWordSpeechGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        self.speeds = [0.75, 1.0, 1.25, 1.5]

    def generate_all_words(self, words):
        """Generate variations for all wake words"""
        for word in words:
            print(f"\n=== Generating audio for '{word}' ===")
            self.generate_variations(word)
        
        print(f"\n✅ Generated audio for {len(words)} wake words: {', '.join(words)}")

    def generate_variations(self, word):
        """Generate audio variations for a single word"""
        output_dir = f"audio_{word}"
        os.makedirs(output_dir, exist_ok=True)
        
        total_variations = len(self.voices) * len(self.speeds)
        
        with tqdm(total=total_variations, desc=f"Generating '{word}' variations") as pbar:
            for voice in self.voices:
                for speed in self.speeds:
                    response = self.client.audio.speech.create(
                        model="tts-1",
                        voice=voice,
                        input=word,
                        speed=speed
                    )
                    
                    filename = f"{output_dir}/{voice}_speed{speed}.mp3"
                    with open(filename, 'wb') as f:
                        f.write(response.content)
                    
                    pbar.update(1)

if __name__ == "__main__":
    generator = MultiWordSpeechGenerator()
    generator.generate_all_words(WORDS) 