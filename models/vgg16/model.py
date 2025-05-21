import torch
import torch.nn as nn
import torchvision.models as models

def create_vgg_model(num_classes, freeze_layers=5):
    """
    Create a VGG16 model with fine-tuning, freezing the specified number of layers.
    
    Args:
        num_classes (int): Number of output classes
        freeze_layers (int): Number of layers to freeze
    
    Returns:
        model (nn.Module): The fine-tuned VGG model
    """
    # Load pre-trained VGG16 model with updated parameters
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    
    # Freeze the first 'freeze_layers' layers
    features_to_freeze = list(model.features.children())[:freeze_layers]
    for layer in features_to_freeze:
        for param in layer.parameters():
            param.requires_grad = False
    
    # Modify the classifier for our number of classes
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    
    return model 