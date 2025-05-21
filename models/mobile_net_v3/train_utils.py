import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
from tqdm import tqdm

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Bir epoch boyunca modeli eğitir
    
    Args:
        model: Eğitilecek model
        dataloader: Eğitim veri yükleyicisi
        criterion: Kayıp fonksiyonu
        optimizer: Optimizer
        device: Eğitim cihazı (CPU/GPU)
    
    Returns:
        epoch_loss: Ortalama epoch kaybı
        epoch_acc: Ortalama epoch doğruluğu
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Gradyanları sıfırla
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass ve optimize et
        loss.backward()
        optimizer.step()
        
        # İstatistikleri güncelle
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    """
    Bir epoch boyunca modeli değerlendirir
    
    Args:
        model: Değerlendirilecek model
        dataloader: Doğrulama veri yükleyicisi
        criterion: Kayıp fonksiyonu
        device: Değerlendirme cihazı (CPU/GPU)
    
    Returns:
        epoch_loss: Ortalama epoch kaybı
        epoch_acc: Ortalama epoch doğruluğu
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # İstatistikleri güncelle
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def get_all_predictions(model, dataloader, device):
    """
    Tüm veri seti için tahminleri ve gerçek etiketleri toplar
    
    Args:
        model: Değerlendirilecek model
        dataloader: Veri yükleyicisi
        device: Değerlendirme cihazı (CPU/GPU)
    
    Returns:
        all_preds: Tüm tahminler
        all_labels: Tüm gerçek etiketler
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)

def plot_training_results(train_losses, val_losses, train_accs, val_accs, save_dir):
    """
    Eğitim ve doğrulama kayıplarını ve doğruluklarını çizer
    
    Args:
        train_losses: Eğitim kayıpları listesi
        val_losses: Doğrulama kayıpları listesi
        train_accs: Eğitim doğrulukları listesi
        val_accs: Doğrulama doğrulukları listesi
        save_dir: Grafiklerin kaydedileceği dizin
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Kayıp grafiği
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
    plt.close()
    
    # Doğruluk grafiği
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'accuracy_curves.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_dir):
    """
    Confusion matrix çizer
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        class_names: Sınıf isimleri
        save_dir: Grafiğin kaydedileceği dizin
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

def generate_classification_report(y_true, y_pred, class_names, save_dir):
    """
    Sınıflandırma raporunu oluşturur ve kaydeder
    
    Args:
        y_true: Gerçek etiketler
        y_pred: Tahmin edilen etiketler
        class_names: Sınıf isimleri
        save_dir: Raporun kaydedileceği dizin
    """
    report = classification_report(y_true, y_pred, target_names=class_names)
    
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    return report 