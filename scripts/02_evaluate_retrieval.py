import os
import torch
import torch.nn.functional as F

# ---------------------------------------------------------
# 1. Pfade & Daten laden
# ---------------------------------------------------------
EMBED_PATH = "../embeddings/dinov2_embeddings.pt"

if not os.path.exists(EMBED_PATH):
    raise FileNotFoundError(f"Datei {EMBED_PATH} nicht gefunden! Bitte zuerst Script 01 ausführen.")

data = torch.load(EMBED_PATH)

query_emb = data["query_emb"]        # Shape: [num_queries, dim]
query_labels = data["query_labels"]  # Shape: [num_queries]
gallery_emb = data["gallery_emb"]    # Shape: [num_gallery, dim]
gallery_labels = data["gallery_labels"] # Shape: [num_gallery]
label_mapping = data["label_mapping"]

# Reverse Mapping für lesbare Ausgaben (ID -> Klassenname)
id2label = {v: k for k, v in label_mapping.items()}

print("--- Data Summary ---")
print(f"Anzahl Queries: {query_emb.shape[0]}")
print(f"Anzahl Gallery: {gallery_emb.shape[0]}")
print(f"Vektor-Dimension: {query_emb.shape[1]}")
print("-" * 20)

# ---------------------------------------------------------
# 2. Similarity Matrix berechnen (Cosine Similarity)
# ---------------------------------------------------------
# Da die Vektoren in Script 01 bereits L2-normalisiert wurden, 
# entspricht das Matrizenprodukt (Dot Product) genau der Cosine Similarity.
sim_matrix = torch.mm(query_emb, gallery_emb.T)  # Shape: [num_queries, num_gallery]

# ---------------------------------------------------------
# 3. Retrieval Evaluierung (Top-1 Match)
# ---------------------------------------------------------
# Für jede Query suchen wir den Index des ähnlichsten Gallery-Eintrags
top1_indices = torch.argmax(sim_matrix, dim=1)

correct_top1 = 0
total_queries = query_emb.shape[0]

print("\n--- Retrieval Einzel-Ergebnisse ---")
for q_idx in range(total_queries):
    predicted_g_idx = top1_indices[q_idx].item()
    
    q_label_id = query_labels[q_idx].item()
    predicted_g_label_id = gallery_labels[predicted_g_idx].item()
    
    q_label_name = id2label[q_label_id]
    predicted_label_name = id2label[predicted_g_label_id]
    
    similarity_score = sim_matrix[q_idx, predicted_g_idx].item()
    
    is_correct = (q_label_id == predicted_g_label_id)
    if is_correct:
        correct_top1 += 1
        
    status = "✅ MATCH" if is_correct else "❌ MISMATCH"
    print(f"Query {q_idx+1} [{q_label_name}] -> Best Match: Gallery {predicted_g_idx+1} [{predicted_label_name}] | Similarity: {similarity_score:.4f} | {status}")

top1_accuracy = (correct_top1 / total_queries) * 100.0

print("\n" + "=" * 30)
print(f"Top-1 Accuracy: {top1_accuracy:.2f}% ({correct_top1}/{total_queries})")
print("=" * 30)