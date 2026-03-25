import torch # PyTorch library for building and training deep learning models
from PIL import Image # Library for opening and manipulating image files
from torchvision.transforms.functional import to_tensor # Function to convert PIL images or arrays to Tensors
from utils import resize_box_xyxy
from args import args

class ObjDetectionDataset(torch.utils.data.Dataset): # Custom Dataset class inheriting from PyTorch's Dataset
    def __init__(self, df):
        # Initialize the class with a DataFrame and reset the index
        self.df = df.reset_index(drop=True)

    def __len__(self):
        # Return the total number of samples (rows) in the dataset
        return len(self.df)

    def __getitem__(self, idx):
        # --- TODO 1: Retrieve the data row at position idx from the dataframe ---
        row = self.df.iloc[idx]
        
        # Open image file from the path in the "images" column and convert to RGB color mode
        img = Image.open(row["images"]).convert("RGB")
        
        # Get the original width (w) and height (h) of the image
        w, h = img.size
        
        # Resize image to the standard size defined in args.py
        img = img.resize((args.image_size, args.image_size))

        # Convert the resized image to a Tensor (numerical data for AI computation)
        image = to_tensor(img)

        # Initialize empty lists to store bounding box coordinates and class labels
        boxes, labels = [], []
        
        # Open the corresponding label file from the path in the "labels" column
        with open(row["labels"]) as f:
            for line in f:
                # Read each line, split values (class_id, x_center, y_center, width, height)
                cls, xc, yc, bw, bh = map(float, line.split())
                
                # Convert normalized coordinates (0 to 1) to actual pixel coordinates
                # Calculate x1, y1 (top-left) and x2, y2 (bottom-right) of the bounding box
                x1 = (xc - bw/2) * w
                y1 = (yc - bh/2) * h
                x2 = (xc + bw/2) * w
                y2 = (yc + bh/2) * h
                
                # Recalculate coordinates to match the new image size (e.g., 512x512)
                x1, y1, x2, y2 = resize_box_xyxy((x1, y1, x2, y2), w, h, args.image_size, args.image_size)
                
                # Add bounding box coordinates to the list
                boxes.append([x1, y1, x2, y2])
                
                # Add class label to the list (adding 1 to distinguish from background class)
                labels.append(int(cls) + 1)

        # Wrap target data into a dictionary
        target = {
            # Convert bounding box list to a 32-bit float Tensor
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            # Convert label list to a 64-bit integer Tensor
            "labels": torch.tensor(labels, dtype=torch.int64),
            # Store the image index for management purposes
            "image_id": torch.tensor([idx]),
        }
        
        # --- TODO 2: Return the processed data pair: image and target ---
        return image, target