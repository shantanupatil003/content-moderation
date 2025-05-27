import os
import argparse
import json
import pandas as pd
import numpy as np
import torchaudio
import torch
from tqdm import tqdm
import shutil
import re
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess TAPAD dataset for Whisper fine-tuning")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the TAPAD dataset root directory"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./tapad_processed",
        help="Path to save the preprocessed dataset"
    )
    parser.add_argument(
        "--create_transcriptions",
        action="store_true",
        help="Create transcriptions file from filenames"
    )
    parser.add_argument(
        "--validate_audio",
        action="store_true",
        help="Validate all audio files and remove corrupted ones"
    )
    parser.add_argument(
        "--normalize_audio",
        action="store_true",
        help="Normalize audio volume"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers for processing"
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=0.1,
        help="Minimum audio duration in seconds to keep"
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=30.0,
        help="Maximum audio duration in seconds to keep"
    )
    return parser.parse_args()

def get_audio_info(audio_path):
    """Get information about audio file."""
    try:
        info = torchaudio.info(audio_path)
        waveform, sample_rate = torchaudio.load(audio_path)
        
        return {
            "path": audio_path,
            "sample_rate": sample_rate,
            "num_channels": waveform.shape[0],
            "num_frames": waveform.shape[1],
            "duration": waveform.shape[1] / sample_rate,
            "valid": True
        }
    except Exception as e:
        logger.warning(f"Error processing {audio_path}: {e}")
        return {
            "path": audio_path,
            "valid": False,
            "error": str(e)
        }

def normalize_audio_file(audio_path, output_path):
    """Normalize audio volume and save to output path."""
    try:
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Normalize volume
        max_val = torch.abs(waveform).max()
        if max_val > 0:
            waveform = waveform / max_val * 0.9  # Leave some headroom
        
        # Save normalized audio
        torchaudio.save(output_path, waveform, sample_rate)
        
        return True
    except Exception as e:
        logger.error(f"Error normalizing {audio_path}: {e}")
        return False

def process_audio_file(args):
    """Process a single audio file with validation and normalization."""
    audio_path, output_path, args = args
    
    try:
        # Get audio information
        info = get_audio_info(audio_path)
        
        if not info["valid"]:
            return {
                "path": audio_path,
                "processed": False,
                "reason": f"Invalid audio file: {info.get('error', 'Unknown error')}"
            }
        
        # Check duration constraints
        if info["duration"] < args.min_duration:
            return {
                "path": audio_path,
                "processed": False,
                "reason": f"Audio too short: {info['duration']:.2f}s < {args.min_duration}s"
            }
        
        if info["duration"] > args.max_duration:
            return {
                "path": audio_path,
                "processed": False,
                "reason": f"Audio too long: {info['duration']:.2f}s > {args.max_duration}s"
            }
        
        # Normalize and save if needed
        if args.normalize_audio:
            success = normalize_audio_file(audio_path, output_path)
            if not success:
                return {
                    "path": audio_path,
                    "processed": False,
                    "reason": "Failed to normalize audio"
                }
        else:
            # Just copy the file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy2(audio_path, output_path)
        
        # Extract transcription from filename
        filename = os.path.basename(audio_path)
        word = os.path.splitext(filename)[0]
        
        return {
            "path": audio_path,
            "output_path": output_path,
            "processed": True,
            "transcription": word,
            "duration": info["duration"],
            "sample_rate": info["sample_rate"]
        }
    
    except Exception as e:
        logger.error(f"Error processing {audio_path}: {e}")
        return {
            "path": audio_path,
            "processed": False,
            "reason": f"Processing error: {str(e)}"
        }

def create_dataset_manifest(processed_files):
    """Create dataset manifest with transcriptions."""
    manifest = {
        "data": []
    }
    
    for file_info in processed_files:
        if file_info["processed"]:
            manifest["data"].append({
                "audio_path": file_info["output_path"],
                "transcription": file_info["transcription"],
                "duration": file_info["duration"]
            })
    
    return manifest

def main():
    """Main preprocessing function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Find all audio files in dataset
    audio_files = []
    for root, _, files in os.walk(args.dataset_path):
        for file in files:
            if file.endswith(".mp3") or file.endswith(".wav"):
                audio_path = os.path.join(root, file)
                
                # Preserve directory structure in output
                rel_path = os.path.relpath(audio_path, args.dataset_path)
                output_path = os.path.join(args.output_path, rel_path)
                
                audio_files.append((audio_path, output_path, args))
    
    logger.info(f"Found {len(audio_files)} audio files in dataset")
    
    # Process audio files
    processed_files = []
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_audio_file, file_args) for file_args in audio_files]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing audio files"):
            result = future.result()
            processed_files.append(result)
    
    # Count successful files
    successful = [f for f in processed_files if f["processed"]]
    logger.info(f"Successfully processed {len(successful)}/{len(audio_files)} files")
    
    # Create dataset manifest
    manifest = create_dataset_manifest(processed_files)
    
    # Save manifest
    manifest_path = os.path.join(args.output_path, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)
    
    # Create transcriptions file (for compatibility with training script)
    if args.create_transcriptions:
        transcriptions = {}
        for file in processed_files:
            if file["processed"]:
                transcriptions[file["output_path"]] = file["transcription"]
        
        transcriptions_path = os.path.join(args.output_path, "transcriptions.json")
        with open(transcriptions_path, "w") as f:
            json.dump(transcriptions, f, indent=4)
    
    # Create train/test split
    df = pd.DataFrame([f for f in processed_files if f["processed"]])
    
    # Group by transcription to ensure we have samples of each word in both train and test
    train_files = []
    test_files = []
    
    for word, group in df.groupby("transcription"):
        # Split 90/10
        split_idx = int(len(group) * 0.9)
        word_train = group.iloc[:split_idx].to_dict("records")
        word_test = group.iloc[split_idx:].to_dict("records")
        
        train_files.extend(word_train)
        test_files.extend(word_test)
    
    # Create splits directory
    splits_dir = os.path.join(args.output_path, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    
    # Save train/test splits
    train_manifest = {"data": [{"audio_path": f["output_path"], "transcription": f["transcription"]} for f in train_files]}
    test_manifest = {"data": [{"audio_path": f["output_path"], "transcription": f["transcription"]} for f in test_files]}
    
    with open(os.path.join(splits_dir, "train.json"), "w") as f:
        json.dump(train_manifest, f, indent=4)
    
    with open(os.path.join(splits_dir, "test.json"), "w") as f:
        json.dump(test_manifest, f, indent=4)
    
    logger.info(f"Created train/test split: {len(train_files)}/{len(test_files)} files")
    logger.info(f"Preprocessing complete. Results saved to {args.output_path}")
    
    # Create summary report
    summary = {
        "total_files": len(audio_files),
        "processed_files": len(successful),
        "failed_files": len(audio_files) - len(successful),
        "train_files": len(train_files),
        "test_files": len(test_files),
        "file_types": {},
        "languages": {}
    }
    
    # Count file types
    for file in successful:
        file_ext = os.path.splitext(file["path"])[1]
        summary["file_types"][file_ext] = summary["file_types"].get(file_ext, 0) + 1
    
    # Count languages
    for file in successful:
        # Extract language from directory structure (assuming language is the parent directory)
        rel_path = os.path.relpath(file["path"], args.dataset_path)
        parts = rel_path.split(os.sep)
        if len(parts) > 1:
            lang = parts[0]
            summary["languages"][lang] = summary["languages"].get(lang, 0) + 1
    
    # Save summary
    with open(os.path.join(args.output_path, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nPreprocessing complete!")
    print(f"- Total files: {summary['total_files']}")
    print(f"- Successfully processed: {summary['processed_files']}")
    print(f"- Failed: {summary['failed_files']}")
    print(f"- Train/Test split: {len(train_files)}/{len(test_files)}")
    
    # Print languages
    print("\nLanguage distribution:")
    for lang, count in sorted(summary["languages"].items(), key=lambda x: x[1], reverse=True):
        print(f"- {lang}: {count} files")

if __name__ == "__main__":
    main()