from args import args
import os
import torch
import torch.optim as optim

def validate_model(model, val_loader, device):
    """
    Function to evaluate the model on the Validation dataset.
    """
    # Set to .train() even during validation to ensure the model returns 
    # Loss values (required for plotting learning curves) instead of predictions.
    model.train() 
    val_loss_sum = 0.0
    val_count = 0
    
    # Disable gradient calculation to save memory and speed up evaluation
    with torch.no_grad():
        for images, targets in val_loader:
            # Move images and targets (boxes, labels) to GPU/CPU and format data types
            images = [image.to(device=device, dtype=torch.float32) for image in images]
            targets = [
                {
                    'boxes': target['boxes'].to(device=device, dtype=torch.float32),
                    'labels': target['labels'].to(device=device, dtype=torch.int64)
                }
                for target in targets
            ]
            
            # Forward pass: get the dictionary of loss values
            loss_dict = model(images, targets)
            # Sum up all loss components (classification, box regression, etc.) into a single scalar
            loss = sum(loss_value for loss_value in loss_dict.values())
            
            # Accumulate loss to calculate the average later
            val_loss_sum += loss.item() * len(images)
            val_count += len(images)

    # Calculate the average loss across the entire validation set
    val_epoch_loss = val_loss_sum / val_count
    return val_epoch_loss

def train_model(model, train_loader, val_loader, device):
    """
    Main function to execute the model training process over multiple Epochs.
    """
    model = model.to(device) # Move the model to the target device (GPU or CPU)
    
    # Initialize the Adam optimizer with parameters defined in args.py
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    best_val_loss = float('inf') # Initialize best loss as infinity for comparison

    for epoch in range(args.epochs):
        model.train() # Set the model to training mode
        running_loss = 0.0
        
        # Iterate through batches provided by the train_loader (linked from main.py)
        for images, targets in train_loader:
            # Prepare data for each Batch (similar to the validation phase)
            images = [image.to(device=device, dtype=torch.float32) for image in images]
            targets = [
                {
                    'boxes': target['boxes'].to(device=device, dtype=torch.float32),
                    'labels': target['labels'].to(device=device, dtype=torch.int64)
                }
                for target in targets
            ]

            optimizer.zero_grad() # Clear gradients from the previous step
            loss_dict = model(images, targets) # Forward pass: Compute losses
            loss = sum(loss_value for loss_value in loss_dict.values())
            
            loss.backward() # Backward pass: Compute gradients (Backpropagation)
            optimizer.step() # Update model weights

            running_loss += loss.item() * len(images)

        # Calculate and display results after each Epoch
        train_epoch_loss = running_loss / len(train_loader.dataset)

        # Execute Validation phase
        val_loss = validate_model(model, val_loader, device)

        # If current performance is better (lower loss), save the model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(args.out_dir, exist_ok=True)
            # Save the model's state_dict to a .pth file
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'best_model.pth'))

        # Print progress to the console
        print(f"Epoch {epoch + 1}/{args.epochs} | "
              f"Train Loss: {train_epoch_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}")