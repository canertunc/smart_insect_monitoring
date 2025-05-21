import time
import torch
import torch.nn as nn
import torch.optim as optim
from config import LEARNING_RATE, NUM_EPOCHS, BEST_MODEL_PATH, LAST_MODEL_PATH
from utils import save_plots, save_confusion_matrix, save_classification_report, get_predictions, print_training_time

def train_model(model, train_loader, val_loader, class_names):
    """
    Modeli eğit
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Kayıp fonksiyonu ve optimize edici tanımla
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    # Eğitim ve doğrulama metriklerini izlemek için listeler
    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []
    
    # En iyi modelin doğrulama doğruluğu
    best_val_acc = 0.0
    
    # Eğitim süresini ölçmek için başlangıç zamanı
    start_time = time.time()
    
    print("\nEğitim başlıyor...")
    print(f"Epoch sayısı: {NUM_EPOCHS}")
    print(f"Öğrenme oranı: {LEARNING_RATE}")
    print("-" * 50)
    
    # Eğitim döngüsü
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        
        # Eğitim modu
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Gradyanları sıfırla
            optimizer.zero_grad()
            
            # İleri yönlü geçiş
            # InceptionV3'ün özelliğinden dolayı eğitim modunda aux_outputs da döndürülür
            outputs, aux_outputs = model(inputs)
            
            # Kayıp hesaplama (hem ana çıktı hem de yardımcı çıktı için)
            loss1 = criterion(outputs, labels)
            loss2 = criterion(aux_outputs, labels)
            loss = loss1 + 0.4 * loss2  # Yardımcı çıktıya daha az ağırlık ver
            
            # Geri yayılım ve optimize etme
            loss.backward()
            optimizer.step()
            
            # İstatistikler
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Epoch eğitim metrikleri
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = 100.0 * correct / total
        train_losses.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)
        
        # Doğrulama modu
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # İleri yönlü geçiş
                outputs = model(inputs)
                
                # Kayıp hesaplama
                val_loss += criterion(outputs, labels).item() * inputs.size(0)
                
                # İstatistikler
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Epoch doğrulama metrikleri
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = 100.0 * val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)
        
        # Epoch süresini hesapla
        epoch_time = time.time() - epoch_start_time
        
        # Epoch sonuçlarını yazdır
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - {epoch_time:.2f} sn")
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%")
        print(f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
        print("-" * 50)
        
        # En iyi modeli kaydet
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"En iyi model kaydedildi: {BEST_MODEL_PATH} (Val Acc: {best_val_acc:.2f}%)")
    
    # Son modeli kaydet
    torch.save(model.state_dict(), LAST_MODEL_PATH)
    print(f"Son model kaydedildi: {LAST_MODEL_PATH}")
    
    # Grafikleri kaydet
    save_plots(train_losses, val_losses, train_acc, val_acc)
    
    # Doğrulama seti üzerinde sınıflandırma raporu ve karışıklık matrisi oluştur
    y_true, y_pred = get_predictions(model, val_loader, device)
    save_classification_report(y_true, y_pred, class_names)
    save_confusion_matrix(y_true, y_pred, class_names)
    
    # Toplam eğitim süresini yazdır
    end_time = time.time()
    print_training_time(start_time, end_time)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'best_val_acc': best_val_acc,
        'training_time': end_time - start_time
    } 