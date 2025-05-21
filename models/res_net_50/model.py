import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet50_model(num_classes, freeze_layers=True):
    """
    ResNet-50 modelini yükler ve fine-tuning için hazırlar.
    
    Args:
        num_classes (int): Sınıflandırma yapılacak sınıf sayısı
        freeze_layers (bool): Eğer True ise, ilk katmanlar dondurulur
        
    Returns:
        model (nn.Module): Fine-tuning için hazırlanmış ResNet-50 modeli
    """
    # Pre-trained ResNet-50 modelini yükle
    model = models.resnet50(pretrained=True)
    
    # İlk katmanları dondur
    if freeze_layers:
        # İlk 6 bloğu dondur (conv1, bn1, relu, maxpool, layer1, layer2)
        freeze_until = [
            model.conv1, 
            model.bn1,
            model.layer1,
            model.layer2
        ]
        
        for module in freeze_until:
            for param in module.parameters():
                param.requires_grad = False
    
    # Son tam bağlantılı katmanı değiştir
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    
    return model 