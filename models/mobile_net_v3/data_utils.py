import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_transforms():
    """
    Eğitim ve doğrulama için veri dönüşümlerini oluşturur
    
    Returns:
        train_transforms: Eğitim için dönüşümler
        val_transforms: Doğrulama için dönüşümler
    """
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms

def load_dataset(data_dir, train_transforms, val_transforms, val_split=0.2, batch_size=32):
    """
    Veri setini yükler ve eğitim/doğrulama bölümlerine ayırır
    
    Args:
        data_dir: Veri setinin bulunduğu klasör
        train_transforms: Eğitim için dönüşümler
        val_transforms: Doğrulama için dönüşümler
        val_split: Doğrulama seti oranı
        batch_size: Batch boyutu
    
    Returns:
        train_loader: Eğitim veri yükleyicisi
        val_loader: Doğrulama veri yükleyicisi
        class_names: Sınıf isimleri listesi
    """
    # Veri setini yükle
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    
    # Sınıf isimlerini al
    class_names = full_dataset.classes
    
    # Veri setini eğitim ve doğrulama olarak böl
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Doğrulama setine doğru dönüşümleri uygula
    val_dataset.dataset = datasets.ImageFolder(root=data_dir, transform=val_transforms)
    
    # DataLoader'ları oluştur
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, class_names 