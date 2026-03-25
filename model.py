# Core PyTorch library for tensor computations (Matrix operations)
import torch

# PyTorch specialized library for Computer Vision
import torchvision

# Import module to modify the final layer (prediction layer) of the Faster R-CNN model
# This is used to declare the number of object classes you want to detect
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Model initialization function. The 'backbone' parameter is a string 
# used to select the base neural network architecture.
def build_model(backbone:str, num_classes:int):
    
    # CASE 1: Select ResNet50 as the backbone
    # This is a large, complex model with high accuracy but high computational cost.
    if backbone == 'fasterrcnn_resnet50_fpn':        
        
        # Load the ResNet50 model architecture. 
        # pretrained=True enables Transfer Learning by downloading weights 
        # pre-trained on millions of images (COCO dataset) instead of starting from scratch.
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        
        # Save the corresponding weights (learned experience) for this model
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights
    
    # CASE 2: Select MobileNet V3 as the backbone
    # This is a lightweight model optimized for high speed (suitable for mobile devices),
    # though accuracy may be slightly lower than ResNet50.
    else:
        
        # Load the MobileNet V3 architecture (also pre-trained with pretrained=True)
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
        
        # Save the weights for MobileNet V3
        weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights
        
    # Get the number of input features for the classifier
    in_feature = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one tailored to your specific number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feature, num_classes)

    return model