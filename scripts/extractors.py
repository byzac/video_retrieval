# Diese Datei schickt die Frames durch das Modell / besteht Wrapper Klassen für DINOv2, ResNet, V-JEPA, etc. 

# Überprüft, Ateipfade (z.B. ob ein Video existiert)
import os

# OpenCV: Liest Videos ein, extrahiert Frames, kann Videos abspielen: Usage for Image Processing, Video Analysis, Object DEtection, Feature Extraction
import cv2

# Troch: Hauptframwork für Tensoren und neuronale Netze
import torch

# Pytroch Standardbibliothek: Bietet fertige Funktionen zur Bildbearbeitung und Data Augmentation (Resize, Convert to TensorSkalieren, Farb-Normalisierung)
import torchvision.transforms as T

# Pillow Standardbibliothek für Bilder, die als Brücke zwischen OpenCV und Pytorch dient: open, edit, save and manipulage images
from PIL import Image 



"""5 Modell-Wrapper Klassen für DINOv2, ResNet, V-JEPA, etc.
    Wenn später ein Objekt aus einer dieser Klassen erstellt wird, (extractor=DINOv2Extractor()),
    führt Python __init__ Methode aus"""
class DINOv2Extractor:
    # Verarbeitet Frame-Tensors (T, C, H, W) -> Mean Pooling -> Vektor (768,)

    
    def __init__(self, device = None):
        """ 1. Damit PyTroch weiß, wo im Speicher die Tensoren und Modelle liegen, wird torch.device genutzt """
         # 1. Prüfen, ob ein DEvice als Argument übergeben wurde
        if device is not None:
            self.device = device
        else:
            # 2. Wenn kein Device übergeben wurde, prüfen, ob Nvidia GPU verfügbar ist
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                # 3. Wenn keine GPU verfügbar ist, auf CPU zurückfallen
                self.device = torch.device("cpu")

        print(f"DinoV2Extractor wird auf Device: {self.device} ausgeführt.")

        """ 2. Modell von PyTorch Hub laden: DINOv2 ist ein vortrainiertes Modell, das auf großen Bilddatensätzen trainiert wurde und visuelle Merkmale extrahieren kann. """
        print("Lade DINOv2 pretrrained Weights (vit_base)...")
        # torch.hub.load(...): Lädt das fertige ViT-Base Modell von Meta inklusive aller gelernten Gewichte herunter. .to(self.device) schiebt das Modell auf die GPU.
        self.model = torch.hub.load("facebookresearch/dinov2:main", "dinov2_vitb14").to(self.device)

        """ 3. Modell in den Evaluierungsmodus schalten """
        # Ein neuronales Netz hat im Trainingsmodus Komponenten wie Dropout oder BatchNormalization, die Zufallselemente enthalten. 
        # .eval() friert das Netz ein, sodass bei gleichem Input immer exakt derselbe Feature-Vektor herauskommt.
        self.model.eval()

        """ 4. Transformations-Pipeline"""
        self.transform = T.Compose([
            T.Resize((224, 224)),  # DINOv2 erwartet 224x224 Pixel große Bilder, resize akaliert also das Bild auf die vom Vision Transformer erwartete Eingabegröße.
            T.ToTensor(),          # Konvertiert ein PIL-Bild (Pixelwerte 0 bis 255) in einen PyTorch-Tensor (Werte 0.0 bis 1.0) und ordnet die Dimensionen um von [Höhe, Breite, Kanäle] zu [Kanäle, Höhe, Breite]
            T.Normalize(           # Subtrahiert den Mittelwert (mean) und teilt durch die Standardabweichung (std) des ImageNet-Datensatzes, um die Zahlenwerte um den Nullpunkt zu zentrieren.
                mean=[0.485, 0.456, 0.406],  # Standardwerte für vortrainierte Modelle
                std=[0.229, 0.224, 0.225]
            )
        ])


    def load_video(self, video_path, num_frames = 16):
        """Liest ein Video mit OpenCV ein und erzeugt einen Tensor aus num_frames Bildern."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video-Datei nicht gefunden: {video_path}")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            raise ValueError(f"Video enthält keine Frames oder konnte nicht gelesen werden: {video_path}")

        # Berechnet num_frames gleichmäßig verteilte Indizes über das gesamte Video
        indices = torch.linspace(0, total_frames - 1, steps=num_frames).long().tolist()

        frames = []
        current_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if current_idx in indices:
               # 1. BGR (OpenCV) -> RGB umwandeln
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 2. In PIL Image konvertieren
                pil_img = Image.fromarray(frame_rgb)
                
                # 3. Transformieren und zur Liste hinzufügen
                frames.append(self.transform(pil_img))

            current_idx += 1

        cap.release()

        # Falls das Video zu kurz war, füllen wir mit dem letzten Frame auf
        while len(frames) < num_frames:
            frames.append(frames[-1])

        # Stapelt die 16 einzelnen Tensors (3, 224, 224) zu einem Tensor der Form (16, 3, 224, 224)
        return torch.stack(frames)


    def extract_embedding(self, video_path, num_frames=16):
        """Erzeugt aus einem Video-Pfad einen 768-dimensionalen Embedding-Vektor."""
        # A) Frames als Tensor laden -> Shape: (16, 3, 224, 224)
        frames_tensor = self.load_video(video_path, num_frames=num_frames).to(self.device)

        # B) Forward Pass ohne Gradienten
        with torch.no_grad():
            # Inferenz pro Frame -> Shape: (16, 768)
            frame_features = self.model(frames_tensor)

            # C) Mean Pooling über die Zeitachse (dim=0) -> Shape: (1, 768)
            video_embedding = frame_features.mean(dim=0, keepdim=True)

            # D) L2-Normalisierung
            video_embedding = torch.nn.functional.normalize(video_embedding, p=2, dim=1)

        return video_embedding

"""""
class ResNet50Extractor:
    # Verarbeitet Frame-Tensors (T, C, H, W) -> Mean Pooling -> Vektor (2048,)
    ...

class SlowFastExtractor:
    # Verarbeitet Clip-Tensors (1, C, T, H, W) -> Vektor (2304,)
    ...

class VJEPAExtractor:
    # Verarbeitet Clip-Tensors (1, C, T, H, W) -> Vektor (768,)
    ...

class DisMoExtractor:
    # Verarbeitet Motion/Appearance Features -> Disentangled Vektor
    ...

    """



# -------------------------------------------------------------------
# Testblock (nur aktiv, wenn extractors.py direkt gestartet wird)
# -------------------------------------------------------------------
if __name__ == "__main__":
    extractor = DINOv2Extractor()
    test_video = "data/test_sample.mp4"
    
    if os.path.exists(test_video):
        embedding = extractor.extract_embedding(test_video)
        print("\n--- TEST ERFOLGREICH ---")
        print(f"Embedding Shape: {embedding.shape}")  # Erwartet: torch.Size([1, 768])
        print(f"L2-Norm: {torch.norm(embedding).item():.4f}")  # Erwartet: 1.0000
    else:
        print(f"\n[Info] Bitte lege ein Test-Video unter '{test_video}' ab, um den Extractor zu testen.")