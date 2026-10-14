# MNIST Veri Seti ile MLP Tabanlı El Yazısı Rakam Tanıma

Bu proje, İstanbul Medeniyet Üniversitesi kapsamında MNIST veri seti kullanılarak bir Multi-Layer Perceptron (MLP) modeli geliştirmek, eğitmek ve farklı hiperparametre yapılandırmalarının performansını karşılaştırmak amacıyla 
gerçekleştirilmiştir. Ayrıca modelin gerçek dünya verileri (öğrenci el yazısı) üzerindeki genelleştirme yeteneği test edilmiştir.

## 📂 Proje Dizin Yapısı

- `handwritten_digits/`
  - `original/`: Kendi el yazımla yazdığım rakam ve öğrenci numarası görselleri.
  - `processed/`: 28x28 gri tonlamalı ve modele uygun hale getirilmiş ön işlenmiş görseller.
- `main.py`: Modelin tanımlandığı, eğitildiği ve test edildiği ana script.
- `predict_handwritten.py` & `predict_student_id.py`: Kılavuz/öğrenci el yazısı tahmin scriptleri.
- `mnist_mlp.pth`: Eğitilmiş PyTorch model dosyası.
- `results_exp1.json`, `results_exp2.json`, `results_exp3.json`: Farklı deney sonuçları.
- `training_curves_exp1.png`, `training_curves_exp2.png`, `training_curves_exp3.png`: Eğitim süreci loss ve accuracy grafikleri.

---

## 🔬 Deneyler ve Hiperparametreler

Projede en az 2 gizli katmanlı (hidden layer) MLP mimarisi kullanılmış, aktivasyon fonksiyonu olarak **ReLU** tercih edilmiştir. Farklı deneylerde öğrenme oranı (learning rate), katman nöron sayıları ve epoch gibi hiperparametreler test edilmiştir:

* **Deney 1 (`results_exp1.json`):** Temel MLP Modeli (Örn: 2 Gizli Katman, SGD/Adam Optimizer)
* **Deney 2 (`results_exp2.json`):** Farklı Nöron Sayısı / Katman Yapılandırması
* **Deney 3 (`results_exp3.json`):** Farklı Learning Rate / Epoch Değişimleri

*(Not: Deney sonuçlarına ait detaylı metrikler ve grafikler repo içerisinde JSON ve PNG formatlarında yer almaktadır.)*

---

## 📊 Eğitim ve Test Sonuçları

Model eğitim süreçlerine ait başarım ve kayıp grafikleri aşağıdadır:
- **Deney 1 Grafiği:** `training_curves_exp1.png`
- **Deney 2 Grafiği:** `training_curves_exp2.png`
- **Deney 3 Grafiği:** `training_curves_exp3.png`

**Raporlanan Temel Metrikler:**
* Training Loss & Validation Loss
* Training Accuracy & Validation Accuracy
* Test Accuracy
* Toplam Eğitim Süresi

---

## ✍️ Kendi El Yazım ve Öğrenci Numarası Test Sonuçları

Model, MNIST eğitim verisinden tamamen bağımsız olarak kendi el yazımla hazırladığım öğrenci numarası ve rakam görselleri üzerinde test edilmiştir. Terminal çıktılarında da görüldüğü üzere model, bazı rakamları yüksek bir güven oranuyla (%99+) doğru tahmin ederken, bazı karakterlerde dağılım kayması (distribution shift) nedeniyle zorlanmıştır.

### Genel Değerlendirme ve Çıkarımlar:
1. **Genelleştirme (Generalization):** MNIST verisindeki font ve çizgi kalınlığı standartları ile gerçek hayattaki el yazım arasındaki farklar, modelin bazı tahminlerde güven oranını düşürmüştür.
2. **Dağılım Kayması (Distribution Shift):** Eğitim verisi ile gerçek dünya verisi arasındaki piksel yoğunluğu ve ortalama farkları modelin performansını doğrudan etkilemektedir. Ön işleme (`processed`) adımlarının önemi bu aşamada açıkça görülmüştür.