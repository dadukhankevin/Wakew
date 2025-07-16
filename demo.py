#!/usr/bin/env python3

from multi_wakeword_detector import MultiWakeWordDetector

class WakeWordDemo:
    def __init__(self):
        try:
            self.detector = MultiWakeWordDetector(
                model_path="./whisper-multi-wake-word",
                callback=self.handle_wake_word,
                verbose=True
            )
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("First: python word_speech_generator.py")
            print("Then: python multi_trainer.py --wake_words 'jeeves,jarvis,alexa,computer' --auto_discover")
            return
        
        self.behaviors = {
            "jeeves": {"emoji": "🤵", "msg": "Jeeves at your service"},
            "jarvis": {"emoji": "🤖", "msg": "JARVIS online"},
            "alexa": {"emoji": "🔵", "msg": "Hello! I'm Alexa"},
            "computer": {"emoji": "💻", "msg": "Computer ready"},
            "assistant": {"emoji": "🎧", "msg": "Assistant activated"}
        }
        
    def handle_wake_word(self, wake_word, transcription):
        behavior = self.behaviors.get(wake_word, {"emoji": "🔊", "msg": f"{wake_word} activated"})
        print(f"\n{behavior['emoji']} {behavior['msg']}")
        print(f"Heard: '{transcription}'\n")

def main():
    demo = WakeWordDemo()
    
    if not hasattr(demo, 'detector'):
        return
    
    print("🎙️ Multi-Wake Word Demo")
    print("Say: jeeves, jarvis, alexa, computer, or assistant")
    print("Press Ctrl+C to stop\n")
    
    try:
        demo.detector.start_listening()
    except KeyboardInterrupt:
        print("\nStopped")

if __name__ == "__main__":
    main() 