import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

class InsectDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_data_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def load_dataset(data_dir='datasetv2', train_ratio=0.8, batch_size=32):
    image_paths = []
    labels = []
    class_names = []
    
    # Sınıf isimlerini al ve sırala
    for class_idx, class_name in enumerate(sorted(os.listdir(data_dir))):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            class_names.append(class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    image_paths.append(img_path)
                    labels.append(class_idx)
    
    # 80-20 olarak train-validation split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, train_size=train_ratio, stratify=labels, random_state=42
    )
    
    # Transformları al
    train_transform, val_transform = get_data_transforms()
    
    # Dataset nesnelerini oluştur
    train_dataset = InsectDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = InsectDataset(val_paths, val_labels, transform=val_transform)
    
    # DataLoader nesnelerini oluştur
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    dataset_info = {
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'num_classes': len(class_names),
        'class_names': class_names
    }
    
    return train_loader, val_loader, dataset_info 