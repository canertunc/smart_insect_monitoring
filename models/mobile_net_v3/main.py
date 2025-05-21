import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
from datetime import datetime
import time

from models import get_mobilenet_v3
from data_utils import get_data_transforms, load_dataset
from train_utils import (train_epoch, validate_epoch, get_all_predictions,
                        plot_training_results, plot_confusion_matrix, 
                        generate_classification_report)

def main():
    # Argümanları ayarla
    parser = argparse.ArgumentParser(description='MobileNetV3 Training for Image Classification')
    parser.add_argument('--data_dir', type=str, default='datasetv2', help='Veri seti dizini')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch boyutu')
    parser.add_argument('--num_epochs', type=int, default=15, help='Epoch sayısı')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Öğrenme oranı')
    parser.add_argument('--fine_tune_layers', type=int, default=3, help='Fine-tune edilecek son katman sayısı')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Ağırlık azaltma')
    
    args = parser.parse_args()
    
    # Çıktı dizinini hazırla
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Argümanları kaydet
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Cihazı ayarla
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Veri dönüşümlerini oluştur
    train_transforms, val_transforms = get_data_transforms()
    
    # Veri setini yükle
    train_loader, val_loader, class_names = load_dataset(
        args.data_dir, 
        train_transforms, 
        val_transforms, 
        val_split=0.2, 
        batch_size=args.batch_size
    )
    
    print(f"Sınıf sayısı: {len(class_names)}")
    print(f"Sınıflar: {class_names}")
    
    # Modeli oluştur
    model = get_mobilenet_v3(num_classes=len(class_names), fine_tune_layers=args.fine_tune_layers)
    model = model.to(device)
    
    # Kayıp fonksiyonu ve optimizer'ı tanımla
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    # Eğitim takibi için değişkenler
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    
    # Eğitim süresini ölçmek için başlangıç zamanını kaydet
    training_start_time = time.time()
    
    # Eğitim döngüsü
    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Eğitim
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Doğrulama
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Sonuçları göster
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Epoch süresini göster
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch süresi: {epoch_time:.2f} saniye")
        
        # Sonuçları kaydet
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Learning rate'i güncelle
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if old_lr != new_lr:
            print(f"Learning rate azaltıldı: {old_lr} -> {new_lr}")
        
        # Son modeli kaydet
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        }, os.path.join(model_dir, f"last_model.pth"))
        
        # En iyi modeli kaydet
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
            }, os.path.join(model_dir, f"best_model.pth"))
            print(f"Yeni en iyi doğruluk: {val_acc:.4f}")
    
    # Toplam eğitim süresini hesapla
    total_training_time = time.time() - training_start_time
    hours, remainder = divmod(total_training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nToplam eğitim süresi: {int(hours)} saat, {int(minutes)} dakika, {seconds:.2f} saniye")
    
    # Eğitim sonuçlarını çiz
    plot_training_results(train_losses, val_losses, train_accs, val_accs, output_dir)
    
    # En iyi modeli yükle ve değerlendir
    checkpoint = torch.load(os.path.join(model_dir, "best_model.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"En iyi model yüklendi (Epoch {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']:.4f})")
    
    # Confusion matrix ve sınıflandırma raporu oluştur
    y_pred, y_true = get_all_predictions(model, val_loader, device)
    plot_confusion_matrix(y_true, y_pred, class_names, output_dir)
    report = generate_classification_report(y_true, y_pred, class_names, output_dir)
    print("\nSınıflandırma Raporu:")
    print(report)
    
    # Eğitim bilgilerini bir dosyaya kaydet
    with open(os.path.join(output_dir, "training_info.txt"), "w") as f:
        f.write(f"Eğitim Parametreleri:\n")
        f.write(f"Batch Boyutu: {args.batch_size}\n")
        f.write(f"Epoch Sayısı: {args.num_epochs}\n")
        f.write(f"Öğrenme Oranı: {args.learning_rate}\n")
        f.write(f"Fine-tune Edilmiş Katman Sayısı: {args.fine_tune_layers}\n\n")
        f.write(f"Eğitim Sonuçları:\n")
        f.write(f"En İyi Doğruluk: {best_val_acc:.4f} (Epoch {checkpoint['epoch']})\n")
        f.write(f"Toplam Eğitim Süresi: {int(hours)} saat, {int(minutes)} dakika, {seconds:.2f} saniye\n")
    
    print(f"\nEğitim tamamlandı! Sonuçlar {output_dir} dizinine kaydedildi.")

if __name__ == "__main__":
    main() 