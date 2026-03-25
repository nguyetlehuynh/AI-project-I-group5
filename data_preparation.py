import os
import csv

# Define data paths
image_dir = "data/images"
label_dir = "data/labels"
csv_output_dir = "data/CSVs"
csv_path = "data/CSVs/dataset.csv"

# Check if the image directory exists
if not os.path.exists(image_dir):
    print(f"Error: Cannot find the directory {image_dir}.")
else:
    # Create the output directory for CSV files if it doesn't exist
    os.makedirs(csv_output_dir, exist_ok=True)
    
    # Get a sorted list of all image files with supported extensions
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    # Create and write to the dataset.csv file
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write the header row
        writer.writerow(['images', 'labels'])
        
        for img in image_files:
            # Get the base filename without extension to match it with the .txt label file
            file_name = os.path.splitext(img)[0]
            label = file_name + ".txt"
            
            # Construct paths and ensure forward slashes are used for cross-platform compatibility
            img_path = os.path.join(image_dir, img).replace("\\", "/")
            label_path = os.path.join(label_dir, label).replace("\\", "/")
            
            # Write the image path and corresponding label path to the CSV
            writer.writerow([img_path, label_path])

    print(f"Successful! File is created in: {csv_path}")