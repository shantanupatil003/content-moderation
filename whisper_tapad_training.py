import os
import sys
import torch
import whisper
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, Dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import TrainingArguments
from pydub import AudioSegment
import subprocess
import re

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

def process_tapad_dataset(tapad_dir):
    """
    Process the TAPAD dataset to create a training dataset for Whisper fine-tuning.
    Args:
        tapad_dir: Path to TAPAD dataset directory
    Returns:
        A Hugging Face dataset ready for fine-tuning
    """
    print("Processing TAPAD dataset...")
    
    audio_files = []
    transcripts = []
    categories = []
    
    # Walk through all directories
    for category in os.listdir(tapad_dir):
        category_path = os.path.join(tapad_dir, category)
        
        # Skip if not a directory or is a hidden directory
        if not os.path.isdir(category_path) or category.startswith('.'):
            continue
            
        print(f"Processing category: {category}")
        
        # Process all MP3 files in this category
        for file in os.listdir(category_path):
            if file.endswith('.mp3'):
                file_path = os.path.join(category_path, file)
                
                # Get the profanity word (category name)
                profanity_word = category
                
                # For training, we need to create a transcript where the profanity word appears
                # This is a simple example - in a real scenario, you might want more context
                transcript = f"This contains the word {profanity_word}."
                
                audio_files.append(file_path)
                transcripts.append(transcript)
                categories.append(profanity_word)
    
    print(f"Collected {len(audio_files)} audio samples across {len(set(categories))} categories")
    
    # Create a Hugging Face dataset
    dataset_dict = {
        "audio": audio_files,
        "text": transcripts,
        "category": categories
    }
    
    # Create a Dataset object
    dataset = Dataset.from_dict(dataset_dict)
    
    # Convert the audio column to the Audio feature
    dataset = dataset.cast_column("audio", Audio())
    
    print("Dataset creation complete")
    return dataset

def prepare_for_training(dataset, output_dir="whisper_tapad_dataset", split_ratio=0.8):
    """
    Prepare the dataset for training by splitting it and saving it
    Args:
        dataset: Hugging Face dataset
        output_dir: Directory to save the processed dataset
        split_ratio: Train/validation split ratio
    Returns:
        Dictionary with train and validation datasets
    """
    print("Preparing dataset for training...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Split the dataset
    train_size = int(len(dataset) * split_ratio)
    val_size = len(dataset) - train_size
    
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, len(dataset)))
    
    print(f"Split dataset into {len(train_dataset)} training and {len(val_dataset)} validation samples")
    
    # Save the datasets
    train_dataset.save_to_disk(os.path.join(output_dir, "train"))
    val_dataset.save_to_disk(os.path.join(output_dir, "validation"))
    
    return {
        "train": train_dataset,
        "validation": val_dataset
    }

def prepare_dataset_for_whisper(example, processor):
    """
    Process a single example for Whisper training
    """
    # Compute log-Mel spectrogram features from input audio
    audio = example["audio"]
    
    # Process audio
    sample_rate = audio["sampling_rate"]
    waveform = audio["array"]
    
    # Convert to correct format for Whisper
    inputs = processor(
        waveform, 
        sampling_rate=sample_rate, 
        return_tensors="pt"
    )
    
    # Process the target text
    target_text = example["text"]
    labels = processor.tokenizer(
        target_text, 
        return_tensors="pt"
    ).input_ids
    
    # Get rid of batch dimension
    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
    
    # Set labels
    inputs["labels"] = labels.squeeze(0)
    
    return inputs

def finetune_whisper(dataset_dict, output_dir="whisper_tapad_model", model_size="base", epochs=3):
    """
    Fine-tune a Whisper model on TAPAD dataset
    Args:
        dataset_dict: Dictionary with train and validation datasets
        output_dir: Directory to save the fine-tuned model
        model_size: Size of the Whisper model to fine-tune
        epochs: Number of training epochs
    Returns:
        Path to the fine-tuned model
    """
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"Loading Whisper {model_size} model...")
    
    # Load Whisper model and processor from Hugging Face
    model_id = f"openai/whisper-{model_size}"
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    
    # Move model to device
    model.to(device)
    
    # Process datasets for training
    def process_dataset(dataset):
        return dataset.map(
            lambda x: prepare_dataset_for_whisper(x, processor),
            remove_columns=dataset.column_names,
            num_proc=4  # Adjust based on your CPU
        )
    
    print("Processing datasets for training...")
    train_dataset = process_dataset(dataset_dict["train"])
    val_dataset = process_dataset(dataset_dict["validation"])
    
    # Configure training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,  # Adjust based on your GPU memory
        per_device_eval_batch_size=8,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="steps",
        save_steps=100,
        learning_rate=1e-5,
        num_train_epochs=epochs,
        weight_decay=0.01,
        fp16=(device == "cuda"),  # Use mixed precision if on GPU
        report_to="none",  # Disable wandb, etc.
        push_to_hub=False,
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.tokenizer
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save the model
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    
    print(f"Model fine-tuned and saved to {output_dir}")
    return output_dir

def extract_profanity_words(tapad_dir):
    """
    Extract profanity words from TAPAD directory structure
    Args:
        tapad_dir: Path to TAPAD dataset
    Returns:
        List of profanity words
    """
    profanity_words = []
    for item in os.listdir(tapad_dir):
        if os.path.isdir(os.path.join(tapad_dir, item)) and not item.startswith('.'):
            profanity_words.append(item)
            
    return profanity_words

def save_profanity_list(profanity_words, output_file="tapad_profanity_list.txt"):
    """
    Save the list of profanity words to a file
    Args:
        profanity_words: List of profanity words
        output_file: Path to output file
    """
    with open(output_file, 'w') as f:
        for word in profanity_words:
            f.write(f"{word}\n")
    
    print(f"Saved {len(profanity_words)} profanity words to {output_file}")

def main():
    # Handle command line arguments
    if len(sys.argv) < 2:
        print("Usage: python train_whisper_tapad.py <command>")
        print("Commands:")
        print("  download           - Download TAPAD dataset")
        print("  process            - Process TAPAD dataset for training")
        print("  train              - Train Whisper on TAPAD dataset")
        print("  all                - Run all steps (download, process, train)")
        print("  extract-words      - Extract profanity word list from TAPAD")
        return
    
    command = sys.argv[1]
    
    if command == "download" or command == "all":
        tapad_dir = download_tapad_dataset()
        if not tapad_dir:
            print("Failed to download TAPAD dataset. Exiting.")
            return
    else:
        tapad_dir = "tapad_dataset"  # Default directory
    
    if command == "process" or command == "all":
        if not os.path.exists(tapad_dir):
            print(f"TAPAD directory {tapad_dir} not found. Run with 'download' first.")
            return
        
        dataset = process_tapad_dataset(tapad_dir)
        dataset_dict = prepare_for_training(dataset)
    
    if command == "train" or command == "all":
        if command == "train" and not os.path.exists("whisper_tapad_dataset/train"):
            print("Processed dataset not found. Run with 'process' first.")
            return
        
        if command == "train":
            # Load the pre-processed dataset
            train_dataset = load_dataset("whisper_tapad_dataset/train", split="train")
            val_dataset = load_dataset("whisper_tapad_dataset/validation", split="validation")
            dataset_dict = {"train": train_dataset, "validation": val_dataset}
        
        # Get model size from argument or use default
        model_size = sys.argv[2] if len(sys.argv) > 2 else "base"
        model_dir = finetune_whisper(dataset_dict, model_size=model_size)
    
    if command == "extract-words" or command == "all":
        if not os.path.exists(tapad_dir):
            print(f"TAPAD directory {tapad_dir} not found. Run with 'download' first.")
            return
        
        profanity_words = extract_profanity_words(tapad_dir)
        save_profanity_list(profanity_words)

if __name__ == "__main__":
    main()