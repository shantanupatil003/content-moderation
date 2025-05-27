# train.py
def train_whisper(model, processor, dataset_root, output_dir, 
                  batch_size=8, learning_rate=5e-5, num_epochs=3):
    """
    Fine-tune the Whisper model on TAPAD dataset.
    
    Args:
        model: Whisper model.
        processor: Whisper processor.
        dataset_root: Path to the TAPAD dataset.
        output_dir: Directory to save the fine-tuned model.
        batch_size: Batch size for training.
        learning_rate: Learning rate for training.
        num_epochs: Number of training epochs.
    """
    # Create dataset and dataloader
    dataset = TAPADDataset(dataset_root, processor)
    
    # Split dataset into train and val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size
    )
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        total_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch to device
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(input_features=input_features, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: loss = {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"  Average training loss: {avg_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_features = batch["input_features"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(input_features=input_features, labels=labels)
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"  Validation loss: {avg_val_loss:.4f}")
        
        model.train()
    
    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")