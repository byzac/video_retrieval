# Recall@K, mAP und Cosine Similarity aus Video-Embeddings berechnen
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

# Hauptverzeichnis und scripts-Ordner zum Pfad hinzufügen
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
SCRIPTS_DIR = FILE.parent

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

import torch
import torch.nn.functional as F

from extractors import DINOv2Extractor, VJEPAExtractor, SlowFastExtractor, ResNet50Extractor

# ---------------------------------------------------------------------------
# Frame-Perturbations (Temporale Störungen für Temporal Sensitivity Tests)
# ---------------------------------------------------------------------------

def apply_perturbation(frames: torch.Tensor, mode: str) -> torch.Tensor:
    """Wendet zeitliche Störungen auf eine Sequenz von Frames an."""
    T = frames.shape[0]
    if mode == "original":
        return frames
    elif mode == "reverse":
        return torch.flip(frames, dims=[0])
    elif mode == "shuffle":
        perm = torch.randperm(T)
        return frames[perm]
    elif mode == "drop_half":
        # Mindestens die Hälfte nehmen, aber darauf achten, dass wir genügend Frames behalten
        indices = torch.arange(0, T, 2)
        return frames[indices]
    else:
        raise ValueError(f"Unbekannter Perturbations-Modus: {mode}")

# ---------------------------------------------------------------------------
# Hilfsfunktionen für Metriken (Recall@K & mAP)
# ---------------------------------------------------------------------------

def compute_metrics(sim_matrix: torch.Tensor):
    """
    Berechnet Recall@1, Recall@5 und mAP aus der N x N Ähnlichkeitsmatrix.
    sim_matrix[i, j] ist die Cosine Sim zwischen Original i und Perturbed j.
    """
    N = sim_matrix.shape[0]
    targets = torch.arange(N, device=sim_matrix.device)

    # Absteigend sortieren -> Top-K Indizes pro Query
    sorted_indices = torch.argsort(sim_matrix, dim=1, descending=True)

    # Recall@1 & Recall@5
    top1_matches = (sorted_indices[:, 0] == targets).float()
    recall_1 = top1_matches.mean().item() * 100.0

    k = min(5, N)
    top5_matches = (sorted_indices[:, :k] == targets.unsqueeze(1)).any(dim=1).float()
    recall_5 = top5_matches.mean().item() * 100.0

    # Mean Average Precision (mAP) für Self-Matching (1 relevantes Ziel pro Query)
    # AP_i = 1 / (Rang der korrekten Zuordnung)
    ranks = (sorted_indices == targets.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1
    map_score = (1.0 / ranks.float()).mean().item() * 100.0

    return recall_1, recall_5, map_score

# ---------------------------------------------------------------------------
# Evaluierung
# ---------------------------------------------------------------------------

def evaluate_retrieval(extractor, video_paths, modes=["original", "reverse", "shuffle", "drop_half"]):
    extractor_name = extractor.__class__.__name__
    print(f"\n==========================================================================")
    print(f" Starte Evaluierung für: {extractor_name} ({len(video_paths)} Videos)")
    print(f"==========================================================================")

    embeddings = {mode: [] for mode in modes}

    for i, video_path in enumerate(video_paths, 1):
        print(f"[{i}/{len(video_paths)}] Verarbeite: {Path(video_path).name}")

        try:
            frames = extractor.load_video(str(video_path))

            for mode in modes:
                perturbed_frames = apply_perturbation(frames, mode)

                with torch.no_grad():
                    emb = extractor.extract_from_frames(perturbed_frames)
                    embeddings[mode].append(emb.cpu())
        except Exception as e:
            print(f"  ❌ Fehler bei {Path(video_path).name}: {e}")
            continue

    if not embeddings["original"]:
        print("Fehler: Keine Video-Embeddings konnten extrahiert werden.")
        return

    for mode in modes:
        embeddings[mode] = torch.cat(embeddings[mode], dim=0)

    orig_emb = embeddings["original"]

    print("\n--------------------------------- EVALUATIONSERGEBNISSE ---------------------------------")
    print(f"{'Perturbation':<12} | {'Mean Cos Sim':<14} | {'Recall@1':<10} | {'Recall@5':<10} | {'mAP':<10}")
    print("-" * 72)

    for mode in modes:
        pert_emb = embeddings[mode]

        # Diagonale Cosine Similarity (Original_i vs Perturbed_i)
        sim_diag = F.cosine_similarity(orig_emb, pert_emb, dim=-1)
        mean_sim = sim_diag.mean().item()

        # N x N Similarity Matrix
        sim_matrix = torch.mm(orig_emb, pert_emb.T)

        # Metriken berechnen
        r1, r5, map_val = compute_metrics(sim_matrix)

        print(f"{mode:<12} | {mean_sim:<14.4f} | {r1:<9.1f}% | {r5:<9.1f}% | {map_val:<9.1f}%")


if __name__ == "__main__":
    # Liste aller Extractor-Klassen, die wir nacheinander evaluieren wollen
    extractor_classes = [
        DINOv2Extractor,
        ResNet50Extractor,
        SlowFastExtractor,
        VJEPAExtractor
    ]

    data_dir = ROOT / "playground_data" / "kinetics400_test"
    video_paths = sorted(list(data_dir.glob("*.mp4")))

    if not video_paths:
        print(f"Keine .mp4 Dateien in {data_dir} gefunden!")
    else:
        for cls in extractor_classes:
            try:
                # Extractor instanziieren
                extractor = cls()
                
                # Evaluierung durchführen
                evaluate_retrieval(extractor, video_paths)
                
                # GPU-Speicher aufräumen für das nächste Modell
                del extractor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n❌ Fehler bei der Evaluierung von {cls.__name__}: {e}")