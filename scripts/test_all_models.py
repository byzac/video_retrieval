import sys
import torch
from pathlib import Path

# Projektverzeichnis zum Pfad hinzufügen
sys.path.append(str(Path(__file__).resolve().parent.parent))

import scripts.extractors as extractors_module

def run_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== STARTE MODELL-HEALTHCHECK AUF DEVICE: {device} ===\n")

    # Erzeuge einen synthetischen Video-Tensor (16 Frames, 3 Kanäle, 224x224)
    dummy_frames = torch.randint(0, 256, (16, 3, 224, 224), dtype=torch.uint8)

    # Verfügbare Extractor-Klassen in scripts/extractors.py ermitteln
    available_classes = [
        attr for attr in dir(extractors_module)
        if attr.endswith("Extractor") and attr != "BaseExtractor"
    ]

    print(f"Gefundene Extractor-Klassen in extractors.py: {available_classes}\n")

    status_summary = {}

    for cls_name in available_classes:
        extractor_cls = getattr(extractors_module, cls_name)
        print("--------------------------------------------------")
        print(f"Testing Extractor: {cls_name}")
        print("--------------------------------------------------")
        try:
            # 1. Instanziierung
            extractor = extractor_cls(device=device)
            
            # 2. Forward Pass mit Dummy Frames
            embedding = extractor.extract_from_frames(dummy_frames)

            # 3. Validation Checks
            assert isinstance(embedding, torch.Tensor), "Output ist kein Tensor!"
            assert embedding.ndim == 2, f"Erwartete Form (1, D), aber erhielt Shape {tuple(embedding.shape)}"
            assert embedding.shape[0] == 1, f"Batch-Dimension muss 1 sein, ist {embedding.shape[0]}"
            assert not torch.isnan(embedding).any(), "Tensor enthält NaN-Werte!"

            print(f"✅ SUCCESS: {cls_name} -> Output Shape: {tuple(embedding.shape)}")
            status_summary[cls_name] = "OK"

        except Exception as e:
            print(f"❌ FAIL: {cls_name} -> Fehler: {str(e)}")
            status_summary[cls_name] = f"ERROR: {str(e)}"
        print("\n")

    print("==================================================")
    print("ZUSAMMENFASSUNG TEST-DURCHLAUF")
    print("==================================================")
    for cls_name, status in status_summary.items():
        print(f"{cls_name:<20} : {status}")

if __name__ == "__main__":
    run_test()