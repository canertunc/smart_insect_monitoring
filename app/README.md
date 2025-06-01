# Böcek Tespit, Sınıflandırma ve Takip Uygulaması

## Türkçe Açıklama

### Proje Hakkında
Bu uygulama, görüntülerdeki ve video akışlarındaki böcekleri tespit eden, sınıflandıran ve takip eden kapsamlı bir sistemdir. Gerçek zamanlı olarak webcam veya video dosyası üzerinden çalışabilir ve tespit edilen böcekler için risk değerlendirmesi sunar.

### Kullanılan Teknolojiler

#### Tespit Modeli: YOLOv8
- Böcekleri tespit etmek için özel eğitilmiş YOLOv8 modeli kullanılmıştır
- Model, görüntülerdeki böceklerin konumunu ve sınıfını yüksek doğrulukla tespit edebilir
- Eşik değeri ayarlanabilir güven skorları ile çalışır

#### Sınıflandırma Modeli: VGG16
- Tespit edilen böceklerin detaylı sınıflandırması için transfer öğrenme ile eğitilmiş VGG16 modeli entegre edilmiştir
- 6 farklı böcek türünü tanımlayabilir: örümcek, lepi (kelebek), çekirge, sinek, arı ve akrep
- Yüksek doğruluklu sınıflandırma sonuçları sunar

#### Nesne Takibi: ByteTrack
- Gelişmiş bir nesne takip algoritması olan ByteTrack özel olarak uygulanmıştır
- Düşük güven skoruna sahip böcekleri bile takip ederek kaybolmalarını engeller
- Kalman filtresi kullanarak hareket tahminlerini optimize eder
- Her böceğe benzersiz ID atar ve hareket geçmişini görselleştirir

#### Kullanıcı Arayüzü
- PyQt5 ile geliştirilmiş kullanıcı dostu arayüz
- Gerçek zamanlı tespit, sınıflandırma ve takip sonuçları
- Ayarlanabilir tespit sıklığı ve güven eşiği
- Risk değerlendirme paneli ve detaylı böcek bilgileri
- Tek fotoğraf sınıflandırma özelliği

### Özellikler

#### Böcek Tespiti
- YOLO modeli ile gerçek zamanlı böcek tespiti
- Ayarlanabilir güven eşiği ile hassasiyeti kontrol etme imkanı
- Her karede veya belirli aralıklarla tespit yapabilme seçeneği

#### Böcek Sınıflandırması
- VGG16 modeli ile tespit edilen böceklerin detaylı sınıflandırması
- 6 farklı böcek türünü ayırt edebilme kapasitesi
- Yüksek doğruluklu tahminler
- Tek bir fotoğraf üzerinde doğrudan sınıflandırma yapabilme

#### Böcek Takibi (ByteTrack)
- Özelleştirilmiş ByteTrack algoritması ile böceklerin takibi
- Düşük eşik değeri ile böceklerin kısa süreli kaybolmaları durumunda bile takip edilebilmesi
- Her böceğe özel ID atama ve hareket geçmişini görselleştirme
- Kalman filtresi ile hareket tahminlerini iyileştirme

#### Risk Değerlendirmesi
- Tespit edilen böcek türlerine bağlı olarak risk değerlendirmesi
- Düşük, orta ve yüksek risk kategorileri
- Her böcek türü için detaylı risk açıklamaları

#### Performans Optimizasyonu
- Seçilebilir tespit frekansı
- Optimum FPS değerleri için ayarlanabilir parametreler
- Takip algoritması sayesinde sürekli tespit ihtiyacını azaltma

#### Tek Fotoğraf Sınıflandırma
Uygulama ayrıca tek bir fotoğraf üzerinde analiz yapma olanağı sağlar:
1. Kullanıcı bir böcek fotoğrafı yükler
2. VGG16 sınıflandırma modeli doğrudan bu fotoğraf üzerinde çalışır
3. Sınıflandırma sonucu ve güven oranı gösterilir
4. Tespit edilen böcek türü için risk değerlendirmesi sunulur

Bu özellik, gerçek zamanlı video analizi yapmadan hızlı sonuçlar elde etmek isteyenler için idealdir.

### Teknik Detaylar ve Özel Uygulamalar

#### YOLOv8 ve VGG16 Birleşimi
Uygulama, iki farklı derin öğrenme modelini birleştirerek çalışır:
1. YOLOv8: Böceklerin görüntüdeki konumunu tespit eder
2. VGG16: Tespit edilen her böceğin daha detaylı sınıflandırmasını yapar

Bu iki aşamalı yaklaşım sayesinde, hem hızlı tespit hem de doğru sınıflandırma mümkün olmaktadır.

#### ByteTrack Entegrasyonu
ByteTrack algoritması, projeye özel olarak uygulanmış ve böcek takibi için optimize edilmiştir:

1. Yüksek Güvenli Tespitler: Öncelikle yüksek güven skorlu tespitler ile eşleştirme yapılır
2. Düşük Güvenli Tespitler: Eşleşmeyen izler, düşük güven skorlu tespitlerle eşleştirilir
3. Kalman Filtresi: Nesne pozisyonlarını ve boyutlarını tahmin etmek için kullanılır
4. Kimlik Yönetimi: Her böceğe benzersiz ID atanır ve korunur

Bu özel entegrasyon, böceklerin kameradan çıksa veya kısa süreliğine tespit edilemese bile takip edilmesini sağlar.

### Kullanım
1. Webcam veya video dosyası seçin
2. Tespit frekansını ve güven eşiğini ayarlayın
3. Gerçek zamanlı tespitleri ve takibi izleyin
4. Tespit edilen böcekler için risk değerlendirmelerini kontrol edin
5. Tek bir fotoğraf analizi için "Classify Image" butonunu kullanın

---

# Insect Detection, Classification and Tracking Application

## English Description

### Project Overview
This application is a comprehensive system for detecting, classifying and tracking insects in images and video streams. It can work in real-time on webcam feeds or video files and provides risk assessment for detected insects.

### Technologies Used

#### Detection Model: YOLOv8
- Custom-trained YOLOv8 model for insect detection
- The model can accurately detect the location and class of insects in images
- Works with adjustable confidence thresholds

#### Classification Model: VGG16
- Transfer learning-trained VGG16 model integrated for detailed classification of detected insects
- Can identify 6 different insect types: spider, lepi (butterfly), grasshopper, fly, bee, and scorpion
- Provides high-accuracy classification results

#### Object Tracking: ByteTrack
- Advanced object tracking algorithm ByteTrack custom implemented
- Tracks insects even with low confidence scores to prevent losing them
- Uses Kalman filtering to optimize motion predictions
- Assigns unique IDs to each insect and visualizes movement history

#### User Interface
- User-friendly interface developed with PyQt5
- Real-time detection, classification, and tracking results
- Adjustable detection frequency and confidence threshold
- Risk assessment panel and detailed insect information
- Single image classification feature

### Features

#### Insect Detection
- Real-time insect detection with YOLO model
- Ability to control sensitivity with adjustable confidence threshold
- Option to perform detection on every frame or at specific intervals

#### Insect Classification
- Detailed classification of detected insects using VGG16 model
- Capacity to distinguish 6 different insect types
- High-accuracy predictions
- Direct classification on a single image

#### Insect Tracking (ByteTrack)
- Tracking of insects with customized ByteTrack algorithm
- Tracking with low threshold values even when insects temporarily disappear
- Unique ID assignment for each insect and movement history visualization
- Improved motion predictions with Kalman filtering

#### Risk Assessment
- Risk assessment based on detected insect types
- Low, medium, and high risk categories
- Detailed risk descriptions for each insect type

#### Performance Optimization
- Selectable detection frequency
- Adjustable parameters for optimal FPS values
- Reduced need for constant detection thanks to tracking algorithm

#### Single Image Classification
The application also provides the ability to analyze a single image:
1. The user uploads an insect photo
2. The VGG16 classification model works directly on this image
3. Classification result and confidence level are displayed
4. Risk assessment is provided for the detected insect type

This feature is ideal for those who want quick results without performing real-time video analysis.

### Technical Details and Custom Implementations

#### YOLOv8 and VGG16 Combination
The application operates by combining two different deep learning models:
1. YOLOv8: Detects the location of insects in the image
2. VGG16: Performs more detailed classification of each detected insect

This two-stage approach enables both fast detection and accurate classification.

#### ByteTrack Integration
The ByteTrack algorithm has been specially implemented and optimized for insect tracking:

1. High Confidence Detections: First, matching is performed with high confidence score detections
2. Low Confidence Detections: Unmatched tracks are matched with low confidence score detections
3. Kalman Filter: Used to predict object positions and sizes
4. Identity Management: Unique IDs are assigned to each insect and preserved

This custom integration ensures that insects are tracked even if they leave the camera view or are temporarily undetectable.

### Usage
1. Select webcam or video file
2. Adjust detection frequency and confidence threshold
3. Monitor real-time detections and tracking
4. Check risk assessments for detected insects
5. Use the "Classify Image" button for single image analysis 