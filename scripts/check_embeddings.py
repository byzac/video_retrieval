import torch

data = torch.load("embeddings/test_dinov2.pt")
print(f"Anzahl gespeicherter Videos: {len(data)}")

# Nimmt den Schlüssel des ersten Videos und zeigt die Form des Tensors
first_key = list(data.keys())[0]
print(f"Beispiel-Video: {first_key}")
print(f"Embedding Shape: {data[first_key].shape}")  # Sollte torch.Size([768]) sein