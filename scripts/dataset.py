# Video Frames laden & Perturbation ausfürhren (Clean, Shuffle, Blur)

import torch
from torch.utils.data import Dataset
import torchvision.io as io
import random

class VideoDataset(Dataset):
    def __init__(self, video_paths, num_frames=16, transform=None, perturbation=None, verbose=True):
        """
        perturbation: None (Original), 'reverse' oder 'shuffle'
        verbose: True gibt Info-Meldungen bei der Initialisierung aus
        """
        self.video_paths = video_paths
        self.num_frames = num_frames
        self.transform = transform
        self.perturbation = perturbation
        self.verbose = verbose

        if self.verbose:
            print(f"[VideoDataset] Initialisiert mit {len(self.video_paths)} Videos.")
            print(f"[VideoDataset] Frame-Abtastung: {self.num_frames} Frames pro Video.")
            print(f"[VideoDataset] Aktiver Perturbations-Modus: '{self.perturbation or 'Original (keine)'}'")

    def __len__(self):
        return len(self.video_paths)

    def _apply_perturbation(self, frames):
        """
        frames: Tensor der Form [T, C, H, W]
        """
        if self.perturbation == 'reverse':
            frames = torch.flip(frames, dims=[0])
        elif self.perturbation == 'shuffle':
            perm = torch.randperm(frames.size(0))
            frames = frames[perm]
        return frames

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        
        # Video laden
        try:
            video, _, _ = io.read_video(video_path, pts_unit='sec')  # [T, H, W, C]
            video = video.permute(0, 3, 1, 2)  # [T, C, H, W]
        except Exception as e:
            print(f"[VideoDataset] Fehler beim Laden von '{video_path}': {e}")
            raise e

        # Uniform Sampling auf num_frames
        total_frames = video.size(0)
        indices = torch.linspace(0, total_frames - 1, self.num_frames).long()
        frames = video[indices]

        # 1. Zeitliche Perturbation anwenden
        if self.perturbation:
            frames = self._apply_perturbation(frames)

        # 2. Räumliche Transformationen (Resize/Norm)
        if self.transform:
            frames = torch.stack([self.transform(frame) for frame in frames])

        return frames, video_path