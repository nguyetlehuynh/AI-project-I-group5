import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import os

# --- CONFIGURATION ---
MODEL_PATH = './sessions/best_model.pth' # Path to your best model
IMAGE_DIR = './test_images/'             # Folder containing 10 new test images
OUTPUT_DIR = './test_results/'           # Folder to save predicted results
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_CLASSES = 2                          # 1 class (Paper) + Background
IMAGE_SIZE = 512
THRESHOLD = 0.5                          # Confidence threshold for visualization

def get_model(num_classes):
    """Initializes the Faster R-CNN model architecture."""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# 1. Load Model Weights
print(f"--- Loading Model from {MODEL_PATH} ---")
model = get_model(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"🚀 Running inference on device: {DEVICE}")

# 2. Perform Inference
with torch.no_grad():
    for img_name in os.listdir(IMAGE_DIR):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(IMAGE_DIR, img_name)
            
            # Pre-processing
            original_img = Image.open(img_path).convert("RGB")
            img_resized = original_img.resize((IMAGE_SIZE, IMAGE_SIZE))
            img_tensor = F.to_tensor(img_resized).unsqueeze(0).to(DEVICE)

            # Prediction
            predictions = model(img_tensor)
            
            # Visualization
            draw = ImageDraw.Draw(img_resized)
            boxes = predictions[0]['boxes'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()

            for box, score in zip(boxes, scores):
                if score > THRESHOLD:
                    # Drawing the bounding box and label
                    draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
                    draw.text((box[0], box[1]), f"Paper Waste: {score:.2f}", fill="red")

            # Saving the result
            img_resized.save(os.path.join(OUTPUT_DIR, img_name))
            print(f"✅ Processed: {img_name}")

print(f"\n🎉 Task Completed! Results saved in: {OUTPUT_DIR}")