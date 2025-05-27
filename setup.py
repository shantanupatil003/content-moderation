# setup.py
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import transformers
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Install required packages if needed
# !pip install transformers datasets torch torchaudio

# Set paths
DATASET_ROOT = "tapad_dataset/audio"  # Update this to your dataset path
OUTPUT_DIR = "whisper_finetuned"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)