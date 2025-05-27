import os
import sys
import time
import json
import shutil
import subprocess
import numpy as np
import pandas as pd
from pydub import AudioSegment
import whisper
import re
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Check for GPU availability and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Helper functions
def process_text(text):
    """Process text by removing punctuation and converting to lowercase."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

def process_audio(audio, time_stamps, output_file="data/audio/censored_output.mp3"):
    """Censor audio at specified timestamps."""
    print(f"Processing audio with {len(time_stamps)} timestamps...")
    if not time_stamps:
        print("No profanity detected, saving original audio")
        audio.export(output_file, format="mp3")
        return
        
    # Generate beep sound for censoring
    if os.path.exists("sounds/quack.mp3"):
        beep = AudioSegment.from_file("sounds/quack.mp3")
        # Extend the quack sound to make it longer (repeat it multiple times)
        beep = beep * 3  # Repeat the quack sound 3 times to make it longer
    else:
        # Fallback if quack sound is not found
        print("Warning: quack.mp3 not found in sounds directory. Using silent beep instead.")
        beep = AudioSegment.silent(duration=1500)  # longer silent duration (1.5 seconds)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Process audio by replacing profane segments with beep
    censored_audio = AudioSegment.empty()
    last_end = 0
    
    for start, end in time_stamps:
        # Convert milliseconds to seconds
        start_ms = int(start)
        end_ms = int(end)
        
        # Add clean segment
        if start_ms > last_end:
            censored_audio += audio[last_end:start_ms]
        
        # Add beep instead of profane word
        word_duration = end_ms - start_ms
        censored_audio += beep[:word_duration]
        
        last_end = end_ms
    
    # Add remaining clean audio
    if last_end < len(audio):
        censored_audio += audio[last_end:]
    
    # Export censored audio
    censored_audio.export(output_file, format="mp3")
    print(f"Censored audio saved to {output_file}")

# TAPAD Dataset Handler
class TAPADDataset(Dataset):
    """Dataset for loading and processing TAPAD audio files."""
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with TAPAD audio files.
            transform (callable, optional): Optional transform to be applied on audio.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.files = []
        self.labels = []
        
        # Walk through all subdirectories
        for subdir, _, files in os.walk(root_dir):
            if files:
                # The profanity type is the name of the subdirectory
                profanity_type = os.path.basename(subdir)
                
                for file in files:
                    if file.endswith('.mp3'):
                        self.files.append(os.path.join(subdir, file))
                        self.labels.append(profanity_type)
        
        print(f"Loaded {len(self.files)} TAPAD audio files across {len(set(self.labels))} categories")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        audio_path = self.files[idx]
        label = self.labels[idx]
        
        # Load audio file
        try:
            audio = AudioSegment.from_file(audio_path)
            
            # Convert to numpy array for model processing if needed
            if self.transform:
                audio = self.transform(audio)
                
            return {
                'audio': audio,
                'path': audio_path,
                'label': label
            }
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            # Return a placeholder in case of error
            return {
                'audio': None,
                'path': audio_path,
                'label': label,
                'error': str(e)
            }

def download_tapad_dataset(target_dir="tapad_dataset"):
    """
    Download the TAPAD dataset for profanity detection.
    Args:
        target_dir: Directory to store the dataset
    Returns:
        Path to the downloaded dataset
    """
    if os.path.exists(target_dir):
        print(f"Dataset directory {target_dir} already exists. Skipping download.")
        return target_dir
        
    print("Downloading TAPAD dataset...")
    try:
        # Clone the repository
        subprocess.run(
            ["git", "clone", "https://github.com/profanitas/TAPAD.git", target_dir],
            check=True
        )
        print(f"TAPAD dataset repository cloned to {target_dir}")
        return target_dir
    except subprocess.CalledProcessError as e:
        print(f"Error downloading TAPAD dataset: {e}")
        print("You might need to manually download it from: https://github.com/profanitas/TAPAD")
        return None
    except Exception as e:
        print(f"Unexpected error downloading dataset: {e}")
        return None

def extract_tapad_keywords(tapad_dir):
    """
    Extract profanity keywords from TAPAD directory structure.
    Args:
        tapad_dir: Path to TAPAD dataset directory
    Returns:
        Set of profanity keywords
    """
    profanity_words = set()
    
    if not os.path.exists(tapad_dir):
        print(f"TAPAD directory {tapad_dir} does not exist.")
        return profanity_words
    
    # Extract profanity words from directory names
    for item in os.listdir(tapad_dir):
        item_path = os.path.join(tapad_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            # Clean up the word (remove special characters, numbers, etc.)
            word = process_text(item)
            if word:
                profanity_words.add(word)
    
    print(f"Extracted {len(profanity_words)} profanity keywords from TAPAD dataset")
    return profanity_words

def fine_tune_whisper_with_tapad(tapad_dir, output_model_dir="fine_tuned_whisper"):
    """
    Fine-tune a Whisper model with TAPAD dataset for better profanity detection.
    This is a simplified implementation - production use would need more thorough training.
    
    Args:
        tapad_dir: Path to TAPAD dataset
        output_model_dir: Directory to save the fine-tuned model
    Returns:
        Path to the fine-tuned model
    """
    # Load dataset
    dataset = TAPADDataset(tapad_dir)
    
    if len(dataset) == 0:
        print("No TAPAD files found for fine-tuning.")
        return None
    
    # Create data loaders
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # This function is a placeholder for the actual fine-tuning process
    # Full implementation would require using Whisper's API for fine-tuning
    # or a custom training loop with PyTorch
    
    print("Note: Full fine-tuning implementation would require more compute resources and time.")
    print("This function provides a framework for integration but does not perform the actual training.")
    
    # Instead, we'll save a modified version of the model with the enhanced profanity list
    os.makedirs(output_model_dir, exist_ok=True)
    
    # Extract profanity words from TAPAD
    profanity_words = extract_tapad_keywords(tapad_dir)
    
    # Save the profanity list for later use with the model
    profanity_file = os.path.join(output_model_dir, "tapad_profanity_list.json")
    with open(profanity_file, 'w') as f:
        json.dump(list(profanity_words), f, indent=2)
    
    print(f"Saved profanity list to {profanity_file}")
    return output_model_dir

def enhanced_profanity_detection(transcript, profanity_words):
    """
    Enhanced profanity detection using the TAPAD profanity word list.
    
    Args:
        transcript: Transcript text to check
        profanity_words: Set of profanity words from TAPAD
    Returns:
        List of detected profanities with indices
    """
    detected = []
    processed_transcript = process_text(transcript)
    words = processed_transcript.split()
    
    for i, word in enumerate(words):
        processed_word = process_text(word)
        if processed_word in profanity_words:
            detected.append((i, word))
    
    return detected

# Main script
def main():
    # Check command line arguments for input file
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        audio_file = sys.argv[1]
    else:
        audio_file = "data/audio/BestFriendX.mp3"
    
    print(f"Processing audio file: {audio_file}")
    
    # Download TAPAD dataset and extract profanity words
    tapad_dir = download_tapad_dataset()
    tapad_profanity = extract_tapad_keywords(tapad_dir)
    
    # Load additional profanity words from existing lists
    try:
        if os.path.exists("words_to_block/profanity_en.csv"):
            df = pd.read_csv("words_to_block/profanity_en.csv", usecols=[0], header=None)
            csv_profanity = set(df[0].dropna().str.strip())
            print(f"Loaded {len(csv_profanity)} profane words from CSV")
        elif os.path.exists("words_to_block/Terms-to-Block.csv"):
            csv_profanity = set(np.genfromtxt("words_to_block/Terms-to-Block.csv", 
                                         delimiter=',', dtype=str, encoding=None))
            print(f"Loaded {len(csv_profanity)} profane words from Terms-to-Block.csv")
        else:
            # Default list if files don't exist
            print("Profanity list files not found. Using only TAPAD keywords.")
            csv_profanity = set()
        
        # Combine with TAPAD profanity words
        profane_words = tapad_profanity.union(csv_profanity)
        processed_words = {process_text(word) for word in profane_words}
        print(f"Combined profanity list contains {len(processed_words)} words")
    except Exception as e:
        print(f"Error loading profanity lists: {e}")
        processed_words = tapad_profanity
        print(f"Using only TAPAD profanity list with {len(processed_words)} words")
    
    # Load audio file
    try:
        audio = AudioSegment.from_file(audio_file)
        print(f"Audio loaded: {len(audio)/1000:.2f} seconds")
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return
    
    # Load Whisper model for transcription
    model_size = "base"  # Options: tiny, base, small, medium, large
    print(f"Loading Whisper {model_size} model...")
    model = whisper.load_model(model_size, device=device)
    
    # Transcribe audio
    print("Transcribing audio...")
    start_time = time.time()
    result = model.transcribe(audio_file)
    elapsed = time.time() - start_time
    print(f"Transcription completed in {elapsed:.2f} seconds")
    
    # Get transcription text
    transcript_text = result["text"]
    print("Transcript:")
    print(transcript_text)
    
    # Get word-level timestamps from Whisper
    # Note: Whisper doesn't provide precise word-level timestamps in all models
    # We need to use the segments and approximated timing
    time_stamps = []
    
    if "segments" in result:
        print("Processing segments for profanity...")
        for segment in result["segments"]:
            segment_start = segment["start"] * 1000  # Convert to ms
            segment_end = segment["end"] * 1000      # Convert to ms
            segment_text = segment["text"]
            
            # Split segment into words and check each word for profanity
            words = re.findall(r'\b\w+\b', segment_text)
            segment_duration = segment_end - segment_start
            word_approx_duration = segment_duration / (len(words) if words else 1)
            
            current_pos = segment_start
            for word in words:
                processed_word = process_text(word)
                if processed_word in processed_words:
                    word_start = current_pos
                    word_end = current_pos + word_approx_duration
                    time_stamps.append([word_start, word_end])
                    print(f"Profanity detected: '{word}' at {word_start/1000:.2f}s to {word_end/1000:.2f}s")
                
                current_pos += word_approx_duration
    
    # Process audio to censor profanity
    if time_stamps:
        print(f"Found {len(time_stamps)} instances of profanity")
        process_audio(audio, time_stamps)
    else:
        print("No profanity detected in the audio")
        output_file = "data/audio/clean_output.mp3"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        audio.export(output_file, format="mp3")
        print(f"Original audio saved to {output_file}")

# Model summary and fine-tuning preparation
def print_model_summary(model, model_size="base"):
    """Print a summary of the Whisper model."""
    if not model:
        print("Model not loaded")
        return
        
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Print summary
    print("\n" + "="*50)
    print(f"WHISPER MODEL SUMMARY ({model_size.upper()})")
    print("="*50)
    print(f"Total Parameters:      {total_params:,}")
    print(f"Trainable Parameters:  {trainable_params:,}")
    print(f"Non-trainable:         {total_params - trainable_params:,}")
    
    # Get model structure information
    encoder_dim = model.encoder.conv1.out_channels if hasattr(model.encoder, 'conv1') else "Unknown"
    n_layer = len([m for m in model.modules() if isinstance(m, torch.nn.TransformerEncoderLayer)])
    
    # Try to extract more details about model architecture
    try:
        encoder_layers = sum(1 for m in model.encoder.modules() if isinstance(m, torch.nn.TransformerEncoderLayer))
        decoder_layers = sum(1 for m in model.decoder.modules() if isinstance(m, torch.nn.TransformerDecoderLayer))
        
        # Get info about attention heads if possible
        attn_modules = [m for m in model.modules() if hasattr(m, 'num_heads')]
        if attn_modules:
            n_heads = attn_modules[0].num_heads if hasattr(attn_modules[0], 'num_heads') else "Unknown"
        else:
            n_heads = "Unknown"
    except Exception as e:
        print(f"Warning: Could not extract detailed architecture info: {e}")
        encoder_layers = "Unknown"
        decoder_layers = "Unknown"
        n_heads = "Unknown"
    
    # Print architecture details
    print("\nArchitecture:")
    print(f"- Model Size:          {model_size}")
    print(f"- Total Layers:        {n_layer}")
    print(f"- Encoder Layers:      {encoder_layers}")
    print(f"- Decoder Layers:      {decoder_layers}")
    print(f"- Attention Heads:     {n_heads}")
    print(f"- Encoder Dimension:   {encoder_dim}")
    print("="*50)
    
    # Save model summary to file for reference
    with open("model_summary.json", "w") as f:
        summary = {
            "model_size": model_size,
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "encoder_layers": encoder_layers if isinstance(encoder_layers, int) else "Unknown",
            "decoder_layers": decoder_layers if isinstance(decoder_layers, int) else "Unknown",
            "attention_heads": n_heads if isinstance(n_heads, int) else "Unknown",
            "encoder_dimension": encoder_dim if isinstance(encoder_dim, int) else "Unknown"
        }
        json.dump(summary, f, indent=4)
    print(f"Model summary saved to model_summary.json")

def prepare_fine_tuning_data(data_dir, profanity_file=None, tapad_dir=None):
    """Prepare dataset for fine-tuning."""
    # Load profanity list if provided
    profane_words = None
    if profanity_file and os.path.exists(profanity_file):
        try:
            df = pd.read_csv(profanity_file, usecols=[0], header=None)
            profane_words = set(df[0].dropna().str.strip())
            print(f"Loaded {len(profane_words)} profane words for dataset labeling")
        except Exception as e:
            print(f"Error loading profanity list: {e}")
    
    # Add TAPAD profanity words if available
    if tapad_dir and os.path.exists(tapad_dir):
        tapad_words = extract_tapad_keywords(tapad_dir)
        if profane_words:
            profane_words = profane_words.union(tapad_words)
        else:
            profane_words = tapad_words
        print(f"Combined profanity list now contains {len(profane_words)} words")
    
    # Create dataset
    dataset = AudioTextDataset(data_dir, profane_words)
    
    # Save dataset metadata
    metadata = {
        "dataset_size": len(dataset),
        "data_path": data_dir,
        "samples_with_profanity": sum(1 for sample in dataset.samples if sample['has_profanity']),
        "samples_without_profanity": sum(1 for sample in dataset.samples if not sample['has_profanity'])
    }
    
    with open("fine_tuning_dataset.json", "w") as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Dataset prepared with {len(dataset)} samples")
    print(f"- Samples with profanity: {metadata['samples_with_profanity']}")
    print(f"- Samples without profanity: {metadata['samples_without_profanity']}")
    print("Dataset metadata saved to fine_tuning_dataset.json")
    
    return dataset

# Helper class for creating a fine-tuning dataset
class AudioTextDataset(Dataset):
    """Dataset for fine-tuning Whisper with custom audio and transcripts."""
    def __init__(self, data_path, profanity_list=None):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to directory containing audio files and transcripts
            profanity_list: Optional list of profane words for labeling
        """
        self.data_path = data_path
        self.samples = []
        self.profanity_list = profanity_list
        
        # Find all audio files in directory
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith(('.mp3', '.wav', '.m4a')):
                    audio_path = os.path.join(root, file)
                    
                    # Look for matching transcript file (same name but .txt extension)
                    transcript_path = os.path.splitext(audio_path)[0] + '.txt'
                    if os.path.exists(transcript_path):
                        with open(transcript_path, 'r', encoding='utf-8') as f:
                            transcript = f.read().strip()
                            
                        # Check if profanity exists in transcript
                        has_profanity = False
                        if self.profanity_list:
                            for word in self.profanity_list:
                                if process_text(word) in process_text(transcript):
                                    has_profanity = True
                                    break
                                    
                        self.samples.append({
                            'audio_path': audio_path,
                            'transcript': transcript,
                            'has_profanity': has_profanity
                        })
        
        print(f"Loaded {len(self.samples)} samples for fine-tuning")
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]

if __name__ == "__main__":
    # Check for special commands
    if len(sys.argv) > 1:
        if sys.argv[1] == "--prepare-fine-tuning":
            if len(sys.argv) > 2:
                data_dir = sys.argv[2]
                profanity_file = sys.argv[3] if len(sys.argv) > 3 else "words_to_block/profanity_en.csv"
                tapad_dir = download_tapad_dataset()
                prepare_fine_tuning_data(data_dir, profanity_file, tapad_dir)
            else:
                print("Usage: python script.py --prepare-fine-tuning <data_directory> [profanity_file]")
        elif sys.argv[1] == "--model-summary":
            model_size = sys.argv[2] if len(sys.argv) > 2 else "base"
            print(f"Loading Whisper {model_size} model for summary...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = whisper.load_model(model_size, device=device)
            print_model_summary(model, model_size)
        elif sys.argv[1] == "--download-tapad":
            tapad_dir = download_tapad_dataset()
            if tapad_dir:
                print(f"TAPAD dataset downloaded to {tapad_dir}")
                profanity_words = extract_tapad_keywords(tapad_dir)
                print(f"Extracted {len(profanity_words)} profanity keywords")
        else:
            main()
    else:
        main()