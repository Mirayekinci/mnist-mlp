from pathlib import Path

import torch
from torch import nn
from PIL import Image, ImageOps
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


# 1. Klasör yolları
PROJECT_DIR = Path(__file__).parent
ORIGINAL_DIR = PROJECT_DIR / "handwritten_digits" / "original"
PROCESSED_DIR = PROJECT_DIR / "handwritten_digits" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# 2. MLP model mimarisi
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)


# 3. Eğitilmiş modeli yükle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MLP().to(device)
model_path = PROJECT_DIR / "mnist_mlp.pth"

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


# 4. Görüntüyü modelin beklediği biçime dönüştür
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def preprocess_single_digit(digit_image):
    """Tek bir rakam görselini MNIST standardına (28x28, merkezleme, oran koruma) getirir."""
    bbox = digit_image.getbbox()
    if bbox is None:
        return None
    digit_image = digit_image.crop(bbox)

    width, height = digit_image.width, digit_image.height
    if width > height:
        new_width = 20
        new_height = int(max(1, height * 20 / width))
    else:
        new_height = 20
        new_width = int(max(1, width * 20 / height))

    digit_image = digit_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    x = (28 - digit_image.width) // 2
    y = (28 - digit_image.height) // 2
    canvas.paste(digit_image, (x, y))

    img_array = np.array(canvas, dtype=np.float32)
    cy, cx = ndimage.center_of_mass(img_array)
    
    if not np.isnan(cx) and not np.isnan(cy):
        shift_x = int(np.round(14.0 - cx))
        shift_y = int(np.round(14.0 - cy))
        canvas = Image.fromarray(img_array.astype(np.uint8))
        canvas = canvas.transform(
            (28, 28), 
            Image.AFFINE, 
            (1, 0, -shift_x, 0, 1, -shift_y), 
            resample=Image.Resampling.BILINEAR
        )

    return canvas


def predict_student_number(image_path):
    """Okul numarası fotoğrafını soldan sağa sütun sütun tarayarak rakamları ayırır ve tahmin eder."""
    image = Image.open(image_path).convert("L")
    image = ImageOps.invert(image)

    threshold = 130
    image = image.point(lambda pixel: 255 if pixel > threshold else 0)

    # Dikey sütunlardaki pikselleri toplayarak rakamlar arasındaki boşlukları bulma (Projection profile)
    arr = np.array(image)
    projection = np.sum(arr, axis=0)
    
    # Rakamların başladığı ve bittiği yatay aralıkları (X ekseni sınırlarını) tespit et
    in_digit = False
    digit_boxes = []
    start_x = 0

    for x, val in enumerate(projection):
        if val > 0 and not in_digit:
            in_digit = True
            start_x = max(0, x - 2) # Kenar payı
        elif val == 0 and in_digit:
            in_digit = False
            end_x = min(image.width, x + 2)
            if end_x - start_x > 5:  # Çok küçük gürültüleri ele
                digit_boxes.append((start_x, end_x))

    if in_digit:
        digit_boxes.append((start_x, image.width))

    results = []
    # Her bir rakam kutusunu soldan sağa kesip modele ver
    for i, (xmin, xmax) in enumerate(digit_boxes):
        box = (xmin, 0, xmax, image.height)
        digit_crop = image.crop(box)
        
        canvas = preprocess_single_digit(digit_crop)
        if canvas is None:
            continue

        input_tensor = transform(canvas).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_digit = probabilities.argmax(dim=1).item()
            confidence = probabilities[0, predicted_digit].item() * 100

        results.append((predicted_digit, confidence, canvas))

    return results


# 5. Okul Numarası Dosyasını Test Et
student_num_path = ORIGINAL_DIR / "student_number.jpeg"

if not student_num_path.exists():
    print(f"'{student_num_path}' bulunamadı! Lütfen okul numarası fotoğrafını original klasörüne ekleyin.")
else:
    print("\nOKUL NUMARASI TAHMİN SONUÇLARI (Soldan Sağa):")
    print("-" * 45)

    predictions = predict_student_number(student_num_path)
    
    predicted_number_str = "".join([str(pred[0]) for pred in predictions])
    print(f"Algılanan Okul Numarası: {predicted_number_str}\n")

    for idx, (digit, conf, canvas) in enumerate(predictions):
        print(f"Rakam {idx+1} -> Tahmin: {digit} | Güven: %{conf:.2f}")

    # Görselleştirme
    if predictions:
        fig, axes = plt.subplots(1, len(predictions), figsize=(2 * len(predictions), 3))
        if len(predictions) == 1:
            axes = [axes]

        for ax, (digit, conf, canvas) in zip(axes, predictions):
            ax.imshow(canvas, cmap="gray")
            ax.set_title(f"Tahmin: {digit}\n%{conf:.1f}")
            ax.axis("off")

        plt.suptitle(f"Okul Numarası Okuma: {predicted_number_str}", fontsize=14)
        plt.tight_layout()
        plt.show()