import os
import torch
import random
import numpy as np
from data_loader import load_dataset
from model import get_model
from train import train_model
from config import OUTPUT_DIR

def main():
    # Tekrarlanabilirlik için seed'leri ayarla
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Çıktı klasörünü oluştur
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Veri setini yükle
    train_loader, val_loader, class_names = load_dataset()
    
    # Modeli oluştur
    model = get_model()
    
    # Modeli eğit
    train_results = train_model(model, train_loader, val_loader, class_names)
    
    print("\nEğitim tamamlandı!")
    print(f"En iyi doğrulama doğruluğu: {train_results['best_val_acc']:.2f}%")

if __name__ == "__main__":
    main() 