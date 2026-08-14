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

OUTPUT_EMBED_DIR = "../embeddings"
os.makedirs(OUTPUT_EMBED_DIR, exist_ok=True)

# DINOv2 Modell und Processor laden
MODEL_NAME = "facebook/dinov2-base"
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# ---------------------------------------------------------
# 2. Mini-Dataset laden & Arrow-Tabelle abfragen
# ---------------------------------------------------------
print("Lade Mini-Dataset über Hugging Face...")
dataset = load_dataset("sayakpaul/ucf101-subset", split="train[:60]")

arrow_table = dataset.data.table
column_names = arrow_table.column_names
print(f"Gefundene Spalten in Arrow: {column_names}")

# Dynamische Erkennung der Spalten
video_col_name = "video" if "video" in column_names else column_names[0]

# Label-Spalte finden (sucht nach 'label', '_label', 'labels' oder Fallback auf zweite Spalte)
label_col_name = None
for candidate in ["label", "_label", "labels", "category"]:
    if candidate in column_names:
        label_col_name = candidate
        break

if label_col_name is None and len(column_names) > 1:
    label_col_name = [c for c in column_names if c != video_col_name][0]

print(f"Nutze Video-Spalte: '{video_col_name}' | Label-Spalte: '{label_col_name}'")

def read_video_pyav(video_bytes):
    """Liest Video-Bytes ein und gibt eine Liste von PIL Images zurück."""
    container = av.open(io.BytesIO(video_bytes))
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_image())
    return frames

# ---------------------------------------------------------
# 3. Hilfsfunktion: Video-Frames extrahieren & verarbeiten
# ---------------------------------------------------------
def extract_dinov2_embedding_from_video(video_frames, num_samples=8):
    total_frames = len(video_frames)
    indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
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
# 4. Feature Extraction & Split in Query / Gallery
# ---------------------------------------------------------
query_embeddings = []
query_labels = []
gallery_embeddings = []
gallery_labels = []

print("Extrahiere DINOv2 Embeddings...")

video_column = arrow_table[video_col_name]
label_column = arrow_table[label_col_name] if label_col_name else None
num_samples = len(arrow_table)

for i in range(num_samples):
    # Zugriff auf das Struct-Feld 'bytes'
    video_struct = video_column[i].as_py()
    
    if isinstance(video_struct, dict) and "bytes" in video_struct and video_struct["bytes"] is not None:
        video_bytes = video_struct["bytes"]
    elif isinstance(video_struct, bytes):
        video_bytes = video_struct
    else:
        file_path = video_struct.get("path") if isinstance(video_struct, dict) else None
        if file_path and os.path.exists(file_path):
            with open(file_path, "rb") as f:
                video_bytes = f.read()
        else:
            raise ValueError(f"Konnte keine Video-Bytes für Index {i} finden.")

    # Label auslesen (oder als Index vergeben, falls keine Spalte vorhanden)
    if label_column is not None:
        label = label_column[i].as_py()
    else:
        label = i

    video_frames = read_video_pyav(video_bytes)
    emb = extract_dinov2_embedding_from_video(video_frames)

    # 20% als Query, 80% als Gallery aufteilen
    if i % 5 == 0:
        query_embeddings.append(emb)
        query_labels.append(label)
    else:
        gallery_embeddings.append(emb)
        gallery_labels.append(label)

    if (i + 1) % 10 == 0 or (i + 1) == num_samples:
        print(f"Verarbeitet: {i + 1}/{num_samples} Videos")

# Tensor-Konvertierung
query_embeddings = torch.stack(query_embeddings)
gallery_embeddings = torch.stack(gallery_embeddings)

# Falls Labels Strings/Pfade sind, in Integers mappen
if query_labels and isinstance(query_labels[0], str):
    label_mapping = {l: idx for idx, l in enumerate(set(query_labels + gallery_labels))}
    query_labels = [label_mapping[l] for l in query_labels]
    gallery_labels = [label_mapping[l] for l in gallery_labels]

query_labels = torch.tensor(query_labels)
gallery_labels = torch.tensor(gallery_labels)

# ---------------------------------------------------------
# 5. Speichern
# ---------------------------------------------------------
torch.save({
    "query_emb": query_embeddings,
    "query_labels": query_labels,
    "gallery_emb": gallery_embeddings,
    "gallery_labels": gallery_labels
}, os.path.join(OUTPUT_EMBED_DIR, "dinov2_embeddings.pt"))

print(f"\nFertig! Speicherung erfolgreich in '{OUTPUT_EMBED_DIR}/dinov2_embeddings.pt'")
print(f"Query Shape: {query_embeddings.shape}, Gallery Shape: {gallery_embeddings.shape}")