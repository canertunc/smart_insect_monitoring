# InceptionV3 Sınıflandırma Projesi

Bu proje, PyTorch kullanarak InceptionV3 modelini böcek görüntülerini 6 farklı sınıfa (arı, sinek, çekirge, kelebek, akrep ve örümcek) sınıflandırmak için fine-tune etmektedir.

## Veri Seti

Veri seti 6 sınıftan oluşmaktadır:
- Arı (Bee): 1673 görüntü
- Sinek (Fly): 2067 görüntü
- Çekirge (Grasshopper): 1724 görüntü
- Kelebek (Lepi): 1692 görüntü
- Akrep (Scorpion): 1932 görüntü
- Örümcek (Spider): 1942 görüntü

## Proje Yapısı

Proje modüler bir yapıda tasarlanmıştır:

- `main.py`: Ana uygulama dosyası
- `config.py`: Konfigürasyon ayarları
- `data_loader.py`: Veri yükleme işlemleri
- `model.py`: InceptionV3 modeli sınıfı
- `train.py`: Eğitim mantığı
- `utils.py`: Yardımcı fonksiyonlar
- `requirements.txt`: Gerekli kütüphaneler

## Gereksinimler

Gerekli kütüphaneleri yüklemek için:

```bash
pip install -r requirements.txt
```

## Kullanım

Modeli eğitmek için:

```bash
python main.py
```

## Özellikler

- InceptionV3 modelinin fine-tune edilmesi
- Eğitim ve doğrulama kayıp/doğruluk grafiklerinin kaydedilmesi
- Sınıflandırma raporu ve karışıklık matrisinin oluşturulması
- En iyi modelin ve son modelin kaydedilmesi
- Eğitim süresinin takibi

## Çıktılar

- Eğitilmiş modeller `output/models/` dizinine kaydedilir
- Grafikler `output/plots/` dizinine kaydedilir
- Raporlar `output/reports/` dizinine kaydedilir 