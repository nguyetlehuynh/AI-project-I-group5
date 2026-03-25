from args import get_args  # Import function to fetch parameters from args.py
import pandas as pd        # Library for data manipulation (DataFrames)
from dataset import ObjDetectionDataset
import torch
from torch.utils.data import DataLoader
import os                  # Library for handling file and directory paths
from model import build_model
from trainer import train_model

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    # Initialize args object to retrieve configuration values (paths, batch size, etc.)
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Read data tables (Dataframes)
    # os.path.join ensures cross-platform compatibility for file paths
    # Load training data file
    train_df = pd.read_csv(os.path.join(args.csv_dir, 'train_df.csv'))
    
    # Load validation data file (used for evaluation during training)
    val_df = pd.read_csv(os.path.join(args.csv_dir, 'val_df.csv'))

    # 2. Initialize Dataset and DataLoader objects
    train_dataset = ObjDetectionDataset(train_df)
    val_dataset = ObjDetectionDataset(val_df)

    # 3. Create data loaders
    # num_workers=2 and pin_memory=True optimized for Google Colab/GPU performance
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, 
                              num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                            num_workers=2, pin_memory=torch.cuda.is_available())

    images, targets = next(iter(train_loader))

    print("Success! Got 1 data batch.")
    # Print diagnostic information for the first batch
    print(f"Number of images in this batch: {len(images)} images")
    print(f"Size of the first image (Image number 0): {images[0].shape}")
    print(f"Label structure (target) of the first image: {targets[0]}")

    # 4. Initialize the model
    # Incrementing num_classes by 1 to account for the background class
    model = build_model(args.backbone, num_classes = args.num_classes + 1)

    # 5. Start the training process
    train_model(model, train_loader, val_loader, device)


# Ensure main() only runs when the script is executed directly
if __name__ == '__main__':
    main()