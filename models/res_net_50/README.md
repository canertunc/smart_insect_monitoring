# ResNet-50 ile Böcek Sınıflandırma Projesi

Bu proje, ResNet-50 modeli kullanarak böcek resimlerinin sınıflandırılmasını gerçekleştirir.

## Proje Yapısı

- `dataset.py`: Veri seti işleme ve yükleme fonksiyonları
- `model.py`: ResNet-50 modelinin tanımı ve fine-tuning ayarları
- `train.py`: Eğitim, değerlendirme ve görselleştirme kodları
- `main.py`: Ana çalıştırma dosyası

## Veri Seti

Veri seti `datasetv2` klasöründedir ve aşağıdaki sınıfları içerir:
- spider (örümcek)
- scorpion (akrep)
- none_shadow (gölge yok)
- none_dirt (kir yok)
- none_bg (arka plan yok)
- lepi (lepidoptera - kelebek/güve)
- grasshopper (çekirge)
- fly (sinek)
- bee (arı)

## Eğitim Detayları

- ResNet-50 modelinin ilk katmanları (conv1, bn1, layer1, layer2) dondurulmuştur.
- Model 10 epoch boyunca eğitilir.
- Eğitim/doğrulama oranı: 80%/20%
- Eğitim sonrası en iyi model ve son model kaydedilir.

## Çıktılar

Eğitim sonrasında aşağıdaki çıktılar oluşturulur:
- `results/training_history.png`: Eğitim ve doğrulama kaybı ve doğruluk grafikleri
- `results/confusion_matrix.png`: Karışıklık matrisi
- `results/classification_report.txt`: Sınıflandırma raporu
- `models/best_model.pth`: En yüksek doğruluk değerini veren model
- `models/last_model.pth`: Son epoch'taki model

## Kullanım

Projeyi çalıştırmak için:

```bash
# Gereksinimleri yükleyin
pip install -r requirements.txt

# Eğitimi başlatın
python main.py
``` 