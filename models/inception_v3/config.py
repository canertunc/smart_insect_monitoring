import os

# Veri seti yapılandırması
DATA_DIR = "datasetv2"
TRAIN_RATIO = 0.8
BATCH_SIZE = 32
IMG_SIZE = 299  # InceptionV3 için varsayılan boyut

# Model yapılandırması
NUM_CLASSES = 6
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
FREEZE_LAYERS = 249  # InceptionV3'ün ilk 249 katmanı dondurulacak (fine-tune için)

# Çıktı yapılandırması
OUTPUT_DIR = "output"
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

# Modellerin kaydedileceği dosya yolları
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
LAST_MODEL_PATH = os.path.join(MODEL_DIR, "last_model.pth")

# Çıktı klasörlerini oluştur
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Sınıf isimleri
CLASS_NAMES = ["bee", "fly", "grasshopper", "lepi", "scorpion", "spider"] 