import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from config import PLOT_DIR, REPORT_DIR, CLASS_NAMES

def save_plots(train_losses, val_losses, train_acc, val_acc):
    """
    Eğitim ve doğrulama kayıplarını ve doğruluk grafiklerini kaydet
    """
    # Kayıp grafiği
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Doğruluk grafiği
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'training_curves.png'))
    plt.close()
    
    print(f"Grafikler {PLOT_DIR} klasörüne kaydedildi.")

def save_confusion_matrix(y_true, y_pred, class_names):
    """
    Karışıklık matrisini hesapla ve kaydet
    """
    # Karışıklık matrisini hesapla
    cm = confusion_matrix(y_true, y_pred)
    
    # Görselleştir
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'confusion_matrix.png'))
    plt.close()
    
    print(f"Karışıklık matrisi {PLOT_DIR} klasörüne kaydedildi.")

def save_classification_report(y_true, y_pred, class_names):
    """
    Sınıflandırma raporunu hesapla ve kaydet
    """
    # Sınıflandırma raporunu oluştur
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    
    # Raporu dosyaya kaydet
    report_path = os.path.join(REPORT_DIR, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Sınıflandırma raporu {report_path} dosyasına kaydedildi.")
    
    # Raporu görüntüle
    print("\nSınıflandırma Raporu:")
    print(report)

def get_predictions(model, data_loader, device):
    """
    Veri yükleyicisindeki tüm örnekler için tahminler yap
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Tahmin yap
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # CPU'ya taşı ve listeye ekle
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds)

def print_training_time(start_time, end_time):
    """
    Eğitim süresini hesapla ve yazdır
    """
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    
    print(f"\nToplam eğitim süresi: {time_str} (saat:dakika:saniye)")
    
    # Süreyi dosyaya kaydet
    with open(os.path.join(REPORT_DIR, 'training_time.txt'), 'w') as f:
        f.write(f"Toplam eğitim süresi: {time_str} (saat:dakika:saniye)")
    
    return time_str 