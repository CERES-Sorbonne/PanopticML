# Ajoutez ceci à votre script original, remplacez juste la fonction generate_vectors
import io
import sys
import time
from pathlib import Path

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from tests.siglip_benchmark import SIGLIPTransformer, get_images


def simple_collate_fn(batch):
    """Fonction collate pour éviter les erreurs de sérialisation"""
    images = [item[0] for item in batch]
    paths = [item[1] for item in batch]
    return images, paths


class SimpleImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = torchvision.io.read_image(img_path)
        return image, str(img_path)



def generate_vectors_fixed(transformer, images, batch_size=64, num_workers=8):
    """Version corrigée sans lambda"""

    dataset = SimpleImageDataset(images)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,  # CRUCIAL pour éviter les oscillations
        prefetch_factor=4,  # CRUCIAL pour précharger
        collate_fn=simple_collate_fn,
        drop_last=False
    )

    vectors = []
    image_paths = []

    # Compilation du modèle si possible
    if hasattr(torch, 'compile'):
        try:
            transformer.model = torch.compile(transformer.model, mode="reduce-overhead")
            print("✅ Modèle compilé")
        except:
            print("⚠️ Compilation non disponible")

    print(f"🚀 Pipeline optimisé: batch={batch_size}, workers={num_workers}")

    with tqdm(total=len(images), desc="Processing") as pbar:
        for batch_images, batch_paths in dataloader:

            # Preprocessing
            inputs = transformer.processor(
                images=batch_images,
                return_tensors="pt",
                padding=True
            )

            # Transfer GPU non-blocking (évite les oscillations)
            inputs = {k: v.to(transformer.device, non_blocking=True)
                      for k, v in inputs.items()}

            # Synchronisation pour éviter les race conditions
            if transformer.device != "cpu":
                torch.cuda.synchronize()

            # Inférence GPU
            with torch.no_grad():
                features = transformer.model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)

            # Récupération CPU
            batch_vectors = features.cpu().numpy()

            vectors.extend(batch_vectors)
            image_paths.extend(batch_paths)

            pbar.update(len(batch_images))

    return vectors, image_paths



if __name__ == "__main__":
    folder = r"D:\CorpusImage\documerica\extracted_images"
    siglip = SIGLIPTransformer()
    images = get_images(Path(folder))

    print("Test version corrigée...")
    start_time = time.time()

    vectors, paths = generate_vectors_fixed(siglip, images, batch_size=16, num_workers=4)


    elapsed = time.time() - start_time
    print(f"✅ Terminé en {elapsed:.1f}s")
    print(f"🏃 Vitesse: {len(images) / elapsed:.1f} images/sec")