# Diese Datei schickt die Frames durch das Modell / besteht Wrapper Klassen für DINOv2, ResNet, V-JEPA, etc. 

# Überprüft Dateipfade (z.B. ob ein Video existiert)
import os

import sys

# OpenCV: Liest Videos ein, extrahiert Frames, kann Videos abspielen: Usage for Image Processing, Video Analysis, Object Detection, Feature Extraction
import cv2
import timm

# Torch: Hauptframework für Tensoren und neuronale Netze
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch Standardbibliothek: Bietet fertige Funktionen zur Bildbearbeitung und Data Augmentation (Resize, Convert to Tensor, Skalieren, Farb-Normalisierung)
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights

from transformers import AutoModel, AutoImageProcessor
# Pillow Standardbibliothek für Bilder, die als Brücke zwischen OpenCV und PyTorch dient: open, edit, save and manipulate images
from PIL import Image 


# ---------------------------------------------------------------------------
# Base Extractor Class (Gemeinsame Basisklasse für alle Modelle)
# ---------------------------------------------------------------------------
class BaseExtractor:
    def __init__(self, device=None):
        """ 1. Damit PyTorch weiß, wo im Speicher die Tensoren und Modelle liegen, wird torch.device genutzt """
        # 1. Prüfen, ob ein Device als Argument übergeben wurde
        if device is not None:
            self.device = device
        else:
            # 2. Wenn kein Device übergeben wurde, prüfen, ob Nvidia GPU verfügbar ist
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                # 3. Wenn keine GPU verfügbar ist, auf CPU zurückfallen
                self.device = torch.device("cpu")

    def _sanitize_tensor(self, frames_tensor: torch.Tensor) -> torch.Tensor:
        """Sichert Datentyp, Device und Wertebereich (0.0 - 1.0) ab."""
        if not isinstance(frames_tensor, torch.Tensor):
            frames_tensor = torch.stack(frames_tensor)
        
        frames_tensor = frames_tensor.to(self.device).float()
        
        if frames_tensor.max() > 1.0:
            frames_tensor = frames_tensor / 255.0
            
        return frames_tensor

    def load_video(self, video_path: str, num_frames: int = 16) -> torch.Tensor:
        """Liest ein Video mit OpenCV ein und erzeugt einen Tensor aus num_frames Bildern."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video-Datei nicht gefunden: {video_path}")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
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

    def extract_from_frames(self, frames_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def extract_embedding(self, video_path: str, num_frames: int = 16) -> torch.Tensor:
        """Erzeugt aus einem Video-Pfad einen L2-normalisierten Embedding-Vektor."""
        frames_tensor = self.load_video(video_path, num_frames=num_frames)
        return self.extract_from_frames(frames_tensor)


# ---------------------------------------------------------------------------
# 1. DINOv2 Extractor (2D ViT + Mean Pooling)
# ---------------------------------------------------------------------------
class DINOv2Extractor(BaseExtractor):
    """ Verarbeitet Frame-Tensors (T, C, H, W) -> Mean Pooling -> Vektor (768,) """

    def __init__(self, device=None):
        super().__init__(device)
        print(f"DinoV2Extractor wird auf Device: {self.device} ausgeführt.")

        """ 2. Modell von PyTorch Hub laden: DINOv2 ist ein vortrainiertes Modell, das auf großen Bilddatensätzen trainiert wurde und visuelle Merkmale extrahieren kann. """
        print("Lade DINOv2 pretrained Weights (vit_base)...")
        # torch.hub.load(...): Lädt das fertige ViT-Base Modell von Meta inklusive aller gelernten Gewichte herunter. .to(self.device) schiebt das Modell auf die GPU.
        self.model = torch.hub.load("facebookresearch/dinov2:main", "dinov2_vitb14_reg").to(self.device)

        """ 3. Modell in den Evaluierungsmodus schalten """
        # Ein neuronales Netz hat im Trainingsmodus Komponenten wie Dropout oder BatchNormalization, die Zufallselemente enthalten. 
        # .eval() friert das Netz ein, sodass bei gleichem Input immer exakt derselbe Feature-Vektor herauskommt.
        self.model.eval()

        """ 4. Transformations-Pipeline """
        self.transform = T.Compose([
            T.Resize((224, 224)),  # DINOv2 erwartet 224x224 Pixel große Bilder, resize skaliert also das Bild auf die vom Vision Transformer erwartete Eingabegröße.
            T.ToTensor(),          # Konvertiert ein PIL-Bild (Pixelwerte 0 bis 255) in einen PyTorch-Tensor (Werte 0.0 bis 1.0) und ordnet die Dimensionen um von [Höhe, Breite, Kanäle] zu [Kanäle, Höhe, Breite]
            T.Normalize(           # Subtrahiert den Mittelwert (mean) und teilt durch die Standardabweichung (std) des ImageNet-Datensatzes, um die Zahlenwerte um den Nullpunkt zu zentrieren.
                mean=[0.485, 0.456, 0.406],  # Standardwerte für vortrainierte Modelle
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_from_frames(self, frames_tensor: torch.Tensor) -> torch.Tensor:
        """
        Nimmt einen Frame-Tensor der Form (T, C, H, W) entgegen,
        schickt ihn durch das Modell und gibt ein L2-normalisiertes Embedding zurück.
        """
        frames = self._sanitize_tensor(frames_tensor)

        with torch.no_grad():
            # Inferenz pro Frame -> Shape: (T, 768)
            frame_features = self.model(frames)

            # Mean Pooling über die Zeitachse -> Shape: (1, 768)
            video_embedding = frame_features.mean(dim=0, keepdim=True)

            # L2-Normalisierung
            video_embedding = F.normalize(video_embedding, p=2, dim=-1)

        return video_embedding.float()


# ---------------------------------------------------------------------------
# 2. ResNet50 Extractor (Classic 2D CNN + Mean Pooling)
# ---------------------------------------------------------------------------
class ResNet50Extractor(BaseExtractor):
    """ Verarbeitet Frame-Tensors (T, C, H, W) -> Mean Pooling -> Vektor (2048,) """

    def __init__(self, device=None):
        super().__init__(device)
        print(f"ResNet50Extractor wird auf Device: {self.device} ausgeführt.")

        weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=weights).to(self.device)
        
        # Entferne den Classification Head (fc layer), um die 2048-dim Features vor der Klassifikation zu erhalten
        self.model.fc = nn.Identity()
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_from_frames(self, frames_tensor: torch.Tensor) -> torch.Tensor:
        frames = self._sanitize_tensor(frames_tensor)

        with torch.no_grad():
            # Inferenz pro Frame -> Shape: (T, 2048)
            frame_features = self.model(frames)

            # Mean Pooling über die Zeitachse -> Shape: (1, 2048)
            video_embedding = frame_features.mean(dim=0, keepdim=True)

            # L2-Normalisierung
            video_embedding = F.normalize(video_embedding, p=2, dim=-1)

        return video_embedding.float()


# ---------------------------------------------------------------------------
# 3. V-JEPA Extractor
# ---------------------------------------------------------------------------
class VJEPAExtractor(BaseExtractor):
    def __init__(self, device=None):
        super().__init__(device)
        print(f"VJEPAExtractor auf Device: {self.device}")
        model_name = "facebook/vjepa2-vitl-fpc64-256"
        
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((256, 256), antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.resize = T.Resize((256, 256), antialias=True)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def extract_from_frames(self, frames_tensor: torch.Tensor) -> torch.Tensor:
        frames = self._sanitize_tensor(frames_tensor)
        frames = self.resize(frames)
        
        if torch.abs(frames.mean() - 0.485) > 0.2:
            frames = self.normalize(frames)

        video_tensor = frames.unsqueeze(0).to(self.device)

        with torch.no_grad():
            try:
                outputs = self.model(pixel_values_videos=video_tensor)
            except TypeError:
                try:
                    outputs = self.model(video_tensor)
                except TypeError:
                    outputs = self.model(pixel_values=video_tensor)

            if hasattr(outputs, "last_hidden_state"):
                features = outputs.last_hidden_state
            elif isinstance(outputs, tuple):
                features = outputs[0]
            else:
                features = outputs

            if features.ndim == 3:
                video_embedding = features.mean(dim=1)
            elif features.ndim == 4:
                video_embedding = features.mean(dim=(1, 2))
            else:
                video_embedding = features

            video_embedding = F.normalize(video_embedding, p=2, dim=-1)

        return video_embedding.float()

    
# ---------------------------------------------------------------------------
# 4. SlowFast Extractor (Gefixt)
# ---------------------------------------------------------------------------
class SlowFastExtractor(BaseExtractor):
    def __init__(self, device=None):
        super().__init__(device)
        print(f"SlowFastExtractor auf Device: {self.device}")
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True).to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        ])

    def _pack_pathway_output(self, frames_tensor: torch.Tensor):
        # Shape: [T, C, H, W] -> Benötigt: [1, C, T, H, W]
        inputs = frames_tensor.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)

        # SlowFast erwartet strikt ein Fast-to-Slow Verhältnis von 4:1 (z. B. 32 Fast-Frames, 8 Slow-Frames).
        # Falls die temporale Länge != 32 ist, bringen wir den Tensor per Interpolation exakt auf 32 Frames.
        if inputs.shape[2] != 32:
            inputs = F.interpolate(inputs, size=(32, 224, 224), mode='trilinear', align_corners=False)

        # Fast Pathway: Alle 32 Frames
        fast_pathway = inputs
        # Slow Pathway: Jeder 4. Frame -> Exactly 8 Frames (32 // 4 = 8)
        slow_pathway = inputs[:, :, ::4, :, :]

        return [slow_pathway, fast_pathway]

    def extract_from_frames(self, frames_tensor: torch.Tensor) -> torch.Tensor:
        frames = self._sanitize_tensor(frames_tensor)
        pathway_inputs = self._pack_pathway_output(frames)

        with torch.no_grad():
            features = self.model(pathway_inputs)
            video_embedding = F.normalize(features, p=2, dim=-1)

        return video_embedding.float()
# -------------------------------------------------------------------
# Testblock (nur aktiv, wenn extractors.py direkt gestartet wird)
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("\n--- STARTE EXTRACTOR SMOKE-TEST ---")
    extractor = DINOv2Extractor()

    data_dir = "playground_data/kinetics400_test"

    # Option 1: Echtes Video testen, falls vorhanden
    if os.path.exists(data_dir):
        video_files = [f for f in os.listdir(data_dir) if f.endswith((".mp4", ".avi", ".mkv"))]
        if video_files:
            test_video = os.path.join(data_dir, video_files[0])
            print(f"[Test 1/2] Teste mit lokalem Video: {test_video}")

            embedding = extractor.extract_embedding(test_video)
            print(f"-> Video Embedding Shape: {embedding.shape}")  # Erwartet: torch.Size([1, 768])
            print(f"-> L2-Norm: {torch.norm(embedding).item():.4f}")  # Erwartet: 1.0000

    # Option 2: Direct Frames / Tensor Test (isoliert extract_from_frames)
    print("\n[Test 2/2] Teste mit synthetischen Frames (Tensor Input)...")
    dummy_frames = torch.randn(16, 3, 224, 224)  # 16 Frames, 3 Channels, 224x224
    frame_embedding = extractor.extract_from_frames(dummy_frames)

    print(f"-> Tensor Embedding Shape: {frame_embedding.shape}")  # Erwartet: torch.Size([1, 768])
    print(f"-> L2-Norm: {torch.norm(frame_embedding).item():.4f}")  # Erwartet: 1.0000
    print("\n--- TEST ERFOLGREICH ABGESCHLOSSEN ---")