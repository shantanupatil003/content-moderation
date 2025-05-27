import os
import whisper
import numpy as np
import json

class ProfanityDetector:
    def __init__(self, model_dir, model_size="base"):
        # Load the model
        self.model = whisper.load_model(model_size)
        
        # Load the profanity categories
        with open(os.path.join(model_dir, "profanity_categories.json"), "r") as f:
            self.categories = json.load(f)
        
        print(f"Loaded profanity detector with {len(self.categories)} categories")
    
    def detect_profanity(self, audio_file):
        # Transcribe the audio
        result = self.model.transcribe(audio_file)
        
        # Check for profanity
        detected = []
        for category in self.categories:
            if category.lower() in result["text"].lower():
                detected.append(category)
        
        return {
            "transcript": result["text"],
            "detected_profanity": detected,
            "has_profanity": len(detected) > 0
        }
