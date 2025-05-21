import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from config import DATA_DIR, BATCH_SIZE, TRAIN_RATIO, IMG_SIZE

def get_data_transforms():
    """
    Veri seti için dönüşüm işlemleri (data augmentation kaldırıldı)
    """
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet istatistikleri
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet istatistikleri
    ])
    
    return train_transforms, val_transforms

def load_dataset():
    """
    Veri setini yükle ve eğitim-doğrulama olarak ayır
    """
    train_transforms, val_transforms = get_data_transforms()
    
    # Tüm veri setini yükle
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=train_transforms)
    
    # Eğitim ve doğrulama setleri için boyutları belirle
    dataset_size = len(full_dataset)
    train_size = int(TRAIN_RATIO * dataset_size)
    val_size = dataset_size - train_size
    
    # Veri setini rastgele böl
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Doğrulama seti için transformları değiştir
    val_dataset.dataset.transform = val_transforms
    
    # DataLoader nesnelerini oluştur
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Toplam görüntü sayısı: {dataset_size}")
    print(f"Eğitim seti boyutu: {train_size}")
    print(f"Doğrulama seti boyutu: {val_size}")
    print(f"Sınıflar: {full_dataset.classes}")
    
    return train_loader, val_loader, full_dataset.classes 