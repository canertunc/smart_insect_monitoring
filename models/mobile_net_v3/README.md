# MobileNetV3 Sınıflandırma Projesi

MobileNetV3 modelini kullanarak çoklu sınıf görüntü sınıflandırması yapan bir proje.

## Proje Yapısı

- `models.py`: MobileNetV3 model yapısını ve fine-tuning fonksiyonlarını içerir
- `data_utils.py`: Veri seti yükleme ve işleme fonksiyonlarını içerir
- `train_utils.py`: Eğitim, değerlendirme ve görselleştirme fonksiyonlarını içerir
- `main.py`: Ana eğitim ve değerlendirme kodunu içerir
- `requirements.txt`: Gerekli paketleri listeler

## Kurulum

Gerekli paketleri yüklemek için:

```bash
pip install -r requirements.txt
```

## Kullanım

Modeli varsayılan parametrelerle eğitmek için:

```bash
python main.py
```

Özel parametrelerle eğitmek için:

```bash
python main.py --data_dir datasetv2 --batch_size 32 --num_epochs 15 --learning_rate 0.001 --fine_tune_layers 3
```

### Parametreler

- `--data_dir`: Veri seti dizini (varsayılan: 'datasetv2')
- `--batch_size`: Batch boyutu (varsayılan: 32)
- `--num_epochs`: Epoch sayısı (varsayılan: 15)
- `--learning_rate`: Öğrenme oranı (varsayılan: 0.001)
- `--fine_tune_layers`: Fine-tune edilecek son katman sayısı (varsayılan: 3)
- `--weight_decay`: Ağırlık azaltma (varsayılan: 1e-4)

## Çıktılar

Eğitim sonunda aşağıdaki çıktılar oluşturulur:

- Eğitim ve doğrulama kayıp/doğruluk grafikleri
- Confusion matrix
- Sınıflandırma raporu
- Son epoch ve en iyi doğruluk değerine sahip modeller

Tüm çıktılar, zaman damgalı bir sonuç dizinine kaydedilir (örneğin `results_20231015_123456/`). 