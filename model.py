# model.py
def load_whisper_model(model_name="openai/whisper-small"):
    """
    Load the Whisper model and processor.
    
    Args:
        model_name (str): Name of the Whisper model to load.
    
    Returns:
        tuple: (processor, model)
    """
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    # Move model to device
    model = model.to(device)
    
    return processor, model