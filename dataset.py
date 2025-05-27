# dataset.py
class TAPADDataset(Dataset):
    def __init__(self, root_dir, processor, max_length=30):
        """
        Args:
            root_dir (string): Directory with all the audio files.
            processor (WhisperProcessor): Processor for Whisper model.
            max_length (int): Max length for audio files in seconds.
        """
        self.root_dir = root_dir
        self.processor = processor
        self.max_length = max_length
        
        # Get all audio files recursively
        self.file_paths = []
        self.labels = []
        
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith('.mp3'):
                    file_path = os.path.join(dirpath, filename)
                    self.file_paths.append(file_path)
                    
                    # Extract label from filename (removing the .mp3 extension)
                    label = os.path.splitext(filename)[0]
                    self.labels.append(label)
        
        print(f"Found {len(self.file_paths)} audio files")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample to 16kHz (Whisper's expected sample rate)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        # Convert to numpy array
        waveform = waveform.squeeze().numpy()
        
        # Process inputs
        inputs = self.processor(
            waveform, 
            sampling_rate=sample_rate, 
            return_tensors="pt"
        )
        
        # Process labels
        with self.processor.as_target_processor():
            labels = self.processor(label, return_tensors="pt").input_ids
        
        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        labels = labels.squeeze(0)
        
        return {
            "input_features": inputs.input_features,
            "labels": labels
        }