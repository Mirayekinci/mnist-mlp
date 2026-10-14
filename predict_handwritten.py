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


def prepare_image(image_path):
    # Fotoğrafı gri tonlamaya dönüştür
    image = Image.open(image_path).convert("L")

    # Koyu rakamları beyaz, açık arka planı siyah yap
    image = ImageOps.invert(image)

    # Eşikleme (Thresholding) ile gürültüleri yok et
    threshold = 130
    image = image.point(lambda pixel: 255 if pixel > threshold else 0)

    # Rakamın çevresindeki gereksiz boşlukları kırp
    bbox = image.getbbox()
    if bbox is None:
        return None
    image = image.crop(bbox)

    # En-boy oranını bozmadan en uzun kenarı 20 yapacak şekilde ölçeklendir
    width, height = image.width, image.height
    if width > height:
        new_width = 20
        new_height = int(max(1, height * 20 / width))
    else:
        new_height = 20
        new_width = int(max(1, width * 20 / height))

    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 28x28 siyah bir tuval oluştur
    canvas = Image.new("L", (28, 28), 0)
    
    # Rakamı tuvalin ortasına yerleştir
    x = (28 - image.width) // 2
    y = (28 - image.height) // 2
    canvas.paste(image, (x, y))

    # NumPy'a çevirip kütle merkezine (Center of Mass) göre tam hizala (MNIST standardı)
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


# 5. Fotoğrafları sırayla tahmin et
image_paths = sorted(
    ORIGINAL_DIR.glob("*"),
    key=lambda path: path.stem
)

image_paths = [
    path for path in image_paths
    if path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]
]

if not image_paths:
    print("original klasöründe fotoğraf bulunamadı.")

else:
    print("\nEL YAZISI RAKAM TAHMİNLERİ")
    print("-" * 35)

    results = []

    for image_path in image_paths:
        canvas = prepare_image(image_path)

        if canvas is None:
            print(f"{image_path.name}: Görüntü okunamadı.")
            continue

        # İşlenmiş görüntüyü kaydet
        processed_path = PROCESSED_DIR / (
            image_path.stem + "_processed.png"
        )
        canvas.save(processed_path)

        # Modele giriş olarak ver
        input_tensor = transform(canvas).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)

            predicted_digit = probabilities.argmax(dim=1).item()
            confidence = probabilities[0, predicted_digit].item() * 100

        results.append(
            (image_path.name, predicted_digit, confidence, canvas.copy())
        )

        print(
            f"{image_path.name} -> Tahmin: {predicted_digit} "
            f"| Güven skoru: %{confidence:.2f}"
        )

    # 6. Rakamları ve tahminleri görsel olarak göster
    if results:
        fig, axes = plt.subplots(
            1, len(results),
            figsize=(2 * len(results), 3)
        )

        if len(results) == 1:
            axes = [axes]

        for ax, result in zip(axes, results):
            filename, prediction, confidence, canvas = result

            ax.imshow(canvas, cmap="gray")
            ax.set_title(
                f"{filename}\nTahmin: {prediction}\n%{confidence:.1f}"
            )
            ax.axis("off")

        plt.tight_layout()
        plt.show()