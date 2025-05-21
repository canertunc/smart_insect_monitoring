import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_loaders(data_dir='dataset', batch_size=32, val_split=0.2):
    """
    Create data loaders for training and validation.
    
    Args:
        data_dir (str): Directory containing the dataset
        batch_size (int): Batch size for training
        val_split (float): Proportion of data to use for validation
    
    Returns:
        train_loader, val_loader, class_names
    """
    # Data transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    
    # Get class names
    class_names = full_dataset.classes
    num_classes = len(class_names)
    
    # Split into train and validation sets
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply validation transform to validation dataset
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Dataset loaded with {dataset_size} images across {num_classes} classes")
    print(f"Training set: {train_size} images, Validation set: {val_size} images")
    
    return train_loader, val_loader, class_names, num_classes 