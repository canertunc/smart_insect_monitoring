import torch
import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES, FREEZE_LAYERS

class InceptionV3Model(nn.Module):
    """
    Fine-tune edilmiş InceptionV3 modeli
    """
    def __init__(self):
        super(InceptionV3Model, self).__init__()
        # InceptionV3 modelini önceden eğitilmiş ağırlıklarla yükle
        self.model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        
        # Fine-tuning için bazı katmanları dondur
        for i, param in enumerate(self.model.parameters()):
            if i < FREEZE_LAYERS:  # İlk n katmanı dondur
                param.requires_grad = False
        
        # Son sınıflandırma katmanını hedef sınıf sayısına göre değiştir
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, NUM_CLASSES)
        
    def forward(self, x):
        # Eğitim modunda ise, çıktılar ve yardımcı çıktılar döndürülür
        # Aux_logits, eğitim sırasında yardımcı sınıflandırıcı için kullanılır
        if self.training:
            output, aux_output = self.model(x)
            return output, aux_output
        else:
            # Değerlendirme modunda sadece ana çıktı döndürülür
            return self.model(x)

def get_model():
    """
    InceptionV3 modelini oluştur ve döndür
    """
    model = InceptionV3Model()
    
    # Eğer GPU varsa modeli GPU'ya taşı
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"Model oluşturuldu. Cihaz: {device}")
    
    # Eğitilebilir parametre sayısını hesapla
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Toplam parametre sayısı: {total_params:,}")
    print(f"Eğitilebilir parametre sayısı: {trainable_params:,}")
    print(f"Dondurulmuş parametre sayısı: {total_params - trainable_params:,}")
    
    return model 