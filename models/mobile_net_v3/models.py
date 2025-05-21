import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

def get_mobilenet_v3(num_classes, fine_tune_layers=3):
    """
    MobileNetV3-Small modelini yükler ve fine-tuning için ayarlar
    
    Args:
        num_classes: Sınıflandırılacak sınıf sayısı
        fine_tune_layers: Fine-tune edilecek son katman sayısı
    
    Returns:
        Fine-tune edilmiş MobileNetV3-Small modeli
    """
    # Önceden eğitilmiş MobileNetV3-Small modelini yükle
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    
    # Tüm parametreleri dondur
    for param in model.parameters():
        param.requires_grad = False
    
    # Son birkaç katmanı fine-tune için açalım
    # MobileNetV3'ün features kısmındaki son katmanları açalım
    features_layers = list(model.features)
    for i in range(max(0, len(features_layers) - fine_tune_layers), len(features_layers)):
        for param in features_layers[i].parameters():
            param.requires_grad = True
    
    # Sınıflandırıcı kısmını yeniden tanımla
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features=in_features, out_features=1024),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(in_features=1024, out_features=num_classes)
    )
    
    return model 