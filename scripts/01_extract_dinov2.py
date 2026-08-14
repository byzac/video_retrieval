import os
import io
import av
import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModel

# ---------------------------------------------------------
# 1. Setup & Pfade
# ---------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Nutze Device: {DEVICE}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_EMBED_DIR = os.path.join(PROJECT_DIR, "embeddings")
os.makedirs(OUTPUT_EMBED_DIR, exist_ok=True)

# DINOv2 Modell und Processor laden
MODEL_NAME = "facebook/dinov2-base"
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# ---------------------------------------------------------
# 2. Dataset laden (im Streaming-Modus für alle Videos)
# ---------------------------------------------------------
print("Lade Dataset über Hugging Face...")
# download_mode zwingt HF dazu, die komplette Index-Liste neu abzurufen
dataset = load_dataset("sayakpaul/ucf101-subset", split="train", download_mode="force_redownload")

arrow_table = dataset.data.table
video_column = arrow_table["video"]
num_samples = len(arrow_table)

def parse_class_from_path(file_path):
    """Extrahiert den Klassennamen aus dem Videopfad."""
    filename = os.path.basename(file_path)
    if filename.startswith("v_"):
        parts = filename.split("_")
        if len(parts) >= 2:
            return parts[1]
    return "Unknown"

def read_video_pyav(video_bytes):
    """Liest Video-Bytes ein und gibt eine Liste von PIL Images zurück."""
    container = av.open(io.BytesIO(video_bytes))
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_image())
    return frames

# ---------------------------------------------------------
# 3. Hilfsfunktion: Feature Extraction
# ---------------------------------------------------------
def extract_dinov2_embedding_from_video(video_frames, num_samples=8):
    total_frames = len(video_frames)
    if total_frames == 0:
        raise ValueError("Video enthält keine Frames.")
        
    indices = np.linspace(0, total_frames - 1, min(num_samples, total_frames), dtype=int)
    sampled_frames = [video_frames[i] for i in indices]

    # Preprocessing für DINOv2
    inputs = processor(images=sampled_frames, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        # [CLS]-Token für jeden Frame extrahieren
        frame_embeddings = outputs.last_hidden_state[:, 0, :]
        # Mean Pooling über alle Samples -> 1 Vektor pro Video
        video_embedding = frame_embeddings.mean(dim=0)

    # L2-Normalisierung
    video_embedding = torch.nn.functional.normalize(video_embedding, p=2, dim=0)
    return video_embedding.cpu()

# ---------------------------------------------------------
# 4. Feature Extraction & Query/Gallery Split
# ---------------------------------------------------------
raw_query_embeddings, raw_query_labels = [], []
raw_gallery_embeddings, raw_gallery_labels = [], []

print(f"Insgesamt {num_samples} Videos im PyArrow Table gefunden. Starte Verarbeitung...")

processed_count = 0
for i in range(num_samples):
    # Direkter PyArrow Zugriff auf rohes Dictionary
    video_struct = video_column[i].as_py()
    
    file_path = ""
    video_bytes = None

    if isinstance(video_struct, dict):
        file_path = video_struct.get("path", "")
        video_bytes = video_struct.get("bytes", None)
    
    # Klasse aus Pfad extrahieren
    class_label = parse_class_from_path(file_path)

    if video_bytes is None and file_path and os.path.exists(file_path):
        with open(file_path, "rb") as f:
            video_bytes = f.read()

    if video_bytes is None:
        continue

    try:
        video_frames = read_video_pyav(video_bytes)
        emb = extract_dinov2_embedding_from_video(video_frames)
    except Exception as e:
        print(f"Überspringe Index {i} wegen Fehler: {e}")
        continue

    # 20% Query (jedes 5. Video), 80% Gallery
    if processed_count % 5 == 0:
        raw_query_embeddings.append(emb)
        raw_query_labels.append(class_label)
    else:
        raw_gallery_embeddings.append(emb)
        raw_gallery_labels.append(class_label)

    processed_count += 1
    if processed_count % 5 == 0 or processed_count == num_samples:
        print(f"Erfolgreich verarbeitet: {processed_count}/{num_samples} Videos...")

# Label Mapping von Strings zu Integer-IDs
unique_labels = sorted(list(set(raw_query_labels + raw_gallery_labels)))
label2id = {label: idx for idx, label in enumerate(unique_labels)}

query_embeddings = torch.stack(raw_query_embeddings)
gallery_embeddings = torch.stack(raw_gallery_embeddings)
query_labels = torch.tensor([label2id[l] for l in raw_query_labels])
gallery_labels = torch.tensor([label2id[l] for l in raw_gallery_labels])

# ---------------------------------------------------------
# 5. Speichern
# ---------------------------------------------------------
torch.save({
    "query_emb": query_embeddings,
    "query_labels": query_labels,
    "gallery_emb": gallery_embeddings,
    "gallery_labels": gallery_labels,
    "label_mapping": label2id
}, os.path.join(OUTPUT_EMBED_DIR, "dinov2_embeddings.pt"))

print("\n--- Extraktion erfolgreich abgeschlossen! ---")
print(f"Gespeicherte Vektoren in '{OUTPUT_EMBED_DIR}/dinov2_embeddings.pt'")
print(f"Gefundene eindeutige Klassen ({len(unique_labels)}): {unique_labels}")
print(f"Query Shape: {query_embeddings.shape} | Gallery Shape: {gallery_embeddings.shape}")