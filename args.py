# Parameter Configuration
    # Directories: Paths to data folders.
    # Epochs: Number of training iterations over the full dataset.
    # Fine-tuning parameters: Settings to optimize the model.

import argparse # Library to handle command-line arguments

def get_args():
    # Create the parser object to define program parameters
    parser = argparse.ArgumentParser(description="Model training options")
    parser.add_argument('--backbone', type=str, default='fasterrcnn_resnet_fpn', choices=['fasterrcnn_resnet_fpn'])

    # --- DATA PATH CONFIGURATION ---

    # Number of target classes (excluding background)
    parser.add_argument('--num_classes', type=int, default=1)
    # Image resolution for resizing
    parser.add_argument('--image_size', type=int, default=512)

    # Path to the directory containing CSV data files
    parser.add_argument('--csv_dir', type=str, default='data/CSVs')
    # Path to the directory where results (model, logs) will be saved
    parser.add_argument('--out_dir', type=str, default='./sessions')

    # --- TRAINING PARAMETERS (HYPERPARAMETERS) ---
    # Number of samples processed in one weight update step
    parser.add_argument('--batch_size', type=int, default=8, choices=[8, 16, 32, 64])
    # Total number of training iterations over the entire dataset
    parser.add_argument('--epochs', type=int, default=12)
    
    # --- OPTIMIZATION PARAMETERS ---
    # Learning Rate: Determines the step size for model adjustment in each step
    parser.add_argument('--lr', type=float, default=0.0001)
    # Weight Decay: Regularization technique to prevent overfitting
    parser.add_argument('--wd', type=float, default=1e-4)

    # Return all collected parameters for use in other files
    return parser.parse_args()

# Initialize the 'args' variable from the function output
args = get_args()