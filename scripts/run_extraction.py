import os
import sys
import torch
from tqdm import tqdm

# Fügt das eigene Verzeichnis (scripts/) zum Python-Suchpfad hinzu:
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extractors import DINOv2Extractor

def extract_kinetics_features():
    # 1. Pfade definieren (wie bei dir eingerichtet)
    data_dir = "playground_data/kinetics400_test"
    output_dir = "embeddings"
    output_file = os.path.join(output_dir, "test_dinov2.pt")

    os.makedirs(output_dir, exist_ok=True)

    # 2. Extractor initialisieren
    print("Initialisiere DINOv2 Extractor...")
    extractor = DINOv2Extractor()

    # 3. Überprüfen, ob Datenordner existiert
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Der Ordner '{data_dir}' wurde nicht gefunden!")

    video_files = [f for f in os.listdir(data_dir) if f.endswith(('.mp4', '.avi', '.mkv'))]
    print(f"Gefundene Videos: {len(video_files)}")

    embeddings_db = {}
    failed_videos = []

    # 4. Extraktion starten
    print("\nStarte Feature-Extraktion...")
    for video_name in tqdm(video_files, desc="Extrahiere Embeddings"):
        video_path = os.path.join(data_dir, video_name)

        try:
            embedding = extractor.extract_embedding(video_path, num_frames=16)
            embeddings_db[video_name] = embedding.squeeze(0).cpu()

        except Exception as e:
            failed_videos.append((video_name, str(e)))

    # 5. Speichern
    print(f"\nSpeichere {len(embeddings_db)} Embeddings in '{output_file}'...")
    torch.save(embeddings_db, output_file)

    print("\n--- EXTRAKTION ABGESCHLOSSEN ---")
    print(f"Erfolgreich verarbeitet: {len(embeddings_db)} Videos")
    if failed_videos:
        print(f"Fehlgeschlagen: {len(failed_videos)} Videos")

if __name__ == "__main__":
    extract_kinetics_features()