import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torchvision.transforms as transforms
import torchvision.models as models
from torch import nn
import os
import time

# GPU kullanımını kontrol et
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Cihaz: {device}")

# Load YOLOv8 model for detection
detection_model = YOLO('../weights/yolo/best.pt')

# Load the VGG16 classification model
def load_vgg_model(num_classes, model_path):
    # Model dosyasını yükle
    checkpoint = torch.load(model_path, map_location=device)
    
    # Model yapısını oluştur
    model = models.vgg16(pretrained=False)
    model.classifier[6] = nn.Linear(4096, num_classes)
    
    # 'model_state_dict' anahtarı ile kaydedilmiş modeli yükle
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Sınıf isimlerini dosyadan yükle
    return model, checkpoint['class_names']

# Modeli ve sınıf isimlerini yükle
NUM_CLASSES = 6  # VGG16 modelinizdeki sınıf sayısı 6
classification_model, class_names = load_vgg_model(NUM_CLASSES, '../weights/vgg16/best_vgg_insect_model.pth')

# Preprocessing for VGG16
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def classify_crop(crop_img):
    """Crop edilmiş görüntüyü VGG16 modeli ile sınıflandır"""
    img = Image.fromarray(crop_img)
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        outputs = classification_model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        
    return class_names[predicted.item()]

# Tespit ve sonuçlarını saklamak için
last_results = None 
last_boxes = []
last_labels = []

def process_frame(frame, frame_count, process_every=2):
    """Tek bir kareyi işle: tespit ve sınıflandırma yap"""
    global last_results, last_boxes, last_labels
    
    # Her process_every karede bir tespit yap, diğer karelerde önceden hesaplanmış kutuları kullan
    if frame_count % process_every == 0 or last_results is None:
        # Tespit etme
        results = detection_model(frame)
        last_results = results
        
        # Kutuları ve etiketleri sakla
        last_boxes = []
        last_labels = []
        
        # Her bir tespiti işle
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Güven ve sınıf bilgilerini al
                confidence = box.conf[0].cpu().numpy()
                
                # Minimum güven değeri belirle
                if confidence < 0.5:
                    continue
                    
                # Kutu koordinatlarını al
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Koordinatların çerçeve sınırları içinde olduğundan emin ol
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                cls = int(box.cls[0].cpu().numpy())
                
                # Tespit edilen nesneyi kırp
                crop = frame[y1:y2, x1:x2]
                
                if crop.size == 0:
                    continue
                    
                # Kırpılmış görüntüyü sınıflandır
                class_name = classify_crop(crop)
                
                # Kutu bilgilerini sakla
                last_boxes.append((x1, y1, x2, y2))
                last_labels.append((cls, confidence, class_name))
    
    # Önceden hesaplanmış kutuları çiz
    for i, (x1, y1, x2, y2) in enumerate(last_boxes):
        cls, confidence, class_name = last_labels[i]
        
        # Sınırlayıcı kutuyu çiz
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Tespit güvenini ve sınıflandırma sonucunu göster
        label = f"{detection_model.names[cls]} {confidence:.2f} - {class_name}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def main():
    # Kaynak için kullanıcı girişi al
    source = input("Webcam için 'webcam' yazın veya bir video dosyası yolu girin: ")
    
    # Kullanıcı çıktıyı kaydetmek istiyor mu?
    save_output = input("Çıktı videosunu kaydetmek istiyor musunuz? (e/h): ").lower() == 'e'
    output_writer = None
    
    # Performans parametrelerini kullanıcıdan al
    try:
        process_every = int(input("Her kaç karede bir tespit yapılsın? (1-10 arası, 1=en kaliteli, 10=en hızlı): "))
        if process_every < 1 or process_every > 10:
            process_every = 3  # Varsayılan değer
    except:
        process_every = 3  # Varsayılan değer
        
    display_fps = True
    
    # Video kaynağını başlat
    if source.lower() == 'webcam':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("Hata: Video kaynağı açılamadı.")
        return
    
    # Video özelliklerini al
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # İstenirse çıktı video yazıcısını oluştur
    if save_output:
        if source.lower() == 'webcam':
            output_path = 'webcam_output.mp4'
        else:
            # Giriş dosyasına göre çıktı dosya ismini oluştur
            filename = os.path.basename(source)
            name, ext = os.path.splitext(filename)
            output_path = f"{name}_output{ext}"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Çıktı {output_path} dosyasına kaydediliyor")
    
    # Pencere oluştur
    cv2.namedWindow('Böcek Tespiti ve Sınıflandırması', cv2.WINDOW_NORMAL)
    
    # FPS ölçümü için
    frame_count = 0
    fps_start_time = time.time()
    fps_value = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Kareyi işle
            frame_count += 1
            processed_frame = process_frame(frame, frame_count, process_every)
            
            # FPS hesapla ve göster (her 10 karede bir)
            if frame_count % 10 == 0:
                current_time = time.time()
                elapsed = current_time - fps_start_time
                if elapsed > 0:
                    fps_value = 10 / elapsed
                    fps_start_time = current_time
                
            if display_fps:
                cv2.putText(processed_frame, f"FPS: {fps_value:.1f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            # Tespit durumunu göster
            detect_status = "Tespit: AKTIF" if frame_count % process_every == 0 else "Tespit: pasif"
            cv2.putText(processed_frame, detect_status, (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            # Kareyi göster
            cv2.imshow('Böcek Tespiti ve Sınıflandırması', processed_frame)
            
            # İstenirse kareyi kaydet
            if save_output and output_writer is not None:
                output_writer.write(processed_frame)
            
            # 'q' tuşuna basılırsa çık
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Hata oluştu: {e}")
        print(f"Hata satırı: {e.__traceback__.tb_lineno}")
    
    finally:
        cap.release()
        if output_writer is not None:
            output_writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 