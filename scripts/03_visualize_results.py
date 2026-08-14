import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. Pfade dynamisch bestimmen
# ---------------------------------------------------------
# Ordner des aktuellen Skripts (scripts/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Hauptprojektordner (retrieval_study/)
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

EMBED_PATH = os.path.join(PROJECT_DIR, "embeddings", "dinov2_embeddings.pt")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(EMBED_PATH):
    raise FileNotFoundError(f"Datei {EMBED_PATH} nicht gefunden! Bitte zuerst Script 01 ausführen.")

data = torch.load(EMBED_PATH)

query_emb = data["query_emb"]
query_labels = data["query_labels"]
gallery_emb = data["gallery_emb"]
gallery_labels = data["gallery_labels"]
label_mapping = data["label_mapping"]

# Reverse Mapping für Achsen-Beschriftung (ID -> Klassenname)
id2label = {v: k for k, v in label_mapping.items()}

query_names = [f"Q{i+1}: {id2label[lbl.item()]}" for i, lbl in enumerate(query_labels)]
gallery_names = [f"G{i+1}: {id2label[lbl.item()]}" for i, lbl in enumerate(gallery_labels)]

# ---------------------------------------------------------
# 2. Similarity Matrix berechnen
# ---------------------------------------------------------
sim_matrix = torch.mm(query_emb, gallery_emb.T).numpy()

# ---------------------------------------------------------
# 3. Heatmap erstellen & speichern
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(
    sim_matrix,
    annot=True,
    fmt=".3f",
    cmap="Blues",
    xticklabels=gallery_names,
    yticklabels=query_names,
    cbar_kws={'label': 'Cosine Similarity'}
)

plt.title("DINOv2 Video Retrieval Similarity Matrix", fontsize=14, pad=12)
plt.xlabel("Gallery Videos", fontsize=11)
plt.ylabel("Query Videos", fontsize=11)
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, "similarity_matrix.png")
plt.savefig(output_path, dpi=300)
plt.close()

print(f"✅ Visualisierung erfolgreich gespeichert unter: '{output_path}'")