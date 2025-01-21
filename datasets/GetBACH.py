from datasets import load_dataset

import glob
import os
from PIL import Image

import torchvision.transforms as T
from torch.utils.data import Dataset
import torchstain
import numpy as np

from tqdm import tqdm
from time import time


def create_patches(img_array, patch_size=512, stride=256):
    patches = []
    height, width, _ = img_array.shape
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = img_array[y:y+patch_size, x:x+patch_size, :]
            patches.append(patch)
    return patches


augmentations = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(degrees=15),
    T.RandomAffine(degrees=0, translate=(0.3, 0.3)),
    T.ToTensor()  # Convert PIL Image to a tensor
])


class PatchDataset(Dataset):
        def __init__(self, root_dir, patch_label_map, transform=None):
            self.root_dir = root_dir
            self.patch_label_map = patch_label_map
            self.transform = transform

        def __len__(self):
            return len(self.patch_label_map)

        def __getitem__(self, index):
            patch_filename, label = self.patch_label_map[index]
            patch_path = os.path.join(self.root_dir, patch_filename)
            image = Image.open(patch_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            label_tensor = torch.tensor(label, dtype=torch.long)
            return image, label_tensor


def bach_data(create=False):
    
    output_dir = "datasets/BACH/preprocessed_patches"
    os.makedirs(output_dir, exist_ok=True)

    patch_label_map = []
    patch_map_csv_path = "patch_label_map.csv"
    
    if create:
        ## Get data
        data_path = "datasets/BACH/data"

        train_files = sorted(glob.glob(os.path.join(data_path, "train-*.parquet")))
        test_files = sorted(glob.glob(os.path.join(data_path, "test-*.parquet")))

        data_files = {
            "train": train_files,
            "test": test_files
        }

        raw_datasets = load_dataset("parquet", data_files=data_files)
        train_dataset, test_dataset = raw_datasets['train'], raw_datasets['test']

        ## Stain
        target_path = "datasets/stains/target.png"
        target_img_pil = Image.open(target_path).convert("RGB")

        target_np = np.array(target_img_pil)

        normaliser = torchstain.normalizers.MacenkoNormalizer(backend='numpy')
        normaliser.fit(target_np)

        ## Preprocess data
        idx = 0
        for sample in tqdm(train_dataset, desc="Creating patches from dataset"):
            t = time()
            pil_image = sample["image"]
            label = sample["label"]

            img_np = np.array(pil_image)

            # Macenko stain normalisation
            if img_np.dtype != np.uint8:
                img_np = img_np.astype(np.uint8)
                
            print(time()-t)
            img_norm_np, _, _ = normaliser.normalize(img_np, 255.0)
            img_norm_np = np.clip(img_norm_np, 0, 255).astype(np.uint8)
            print(time()-t)
            # Create 512×512 patches
            patches = create_patches(img_norm_np, patch_size=512, stride=256)
            print(time()-t)
            # Save
            for patch in patches:
                patch_pil = Image.fromarray(patch)
                patch_filename = f"patch_{idx}.png"
                patch_path = os.path.join(output_dir, patch_filename)
                patch_pil.save(patch_path)
                patch_label_map.append((patch_filename, label))
                idx += 1
            
            print(time()-t)
            
        df = pd.DataFrame(patch_label_map, columns=["filename", "label"])
        df.to_csv(patch_map_csv_path, index=False)
        
    else:
        df = pd.read_csv(patch_map_csv_path)
        patch_label_map = list(zip(df["filename"], df["label"]))

    patch_dataset = PatchDataset(root_dir=output_dir, patch_label_map=patch_label_map, transform=augmentations)
    patch_loader = DataLoader(patch_dataset, batch_size=8, shuffle=True, num_workers=2)
    
    return patch_dataset, patch_loader


if __name__ == "__main__":
    patch_dataset, patch_loader = bach_data(create=True)
    for images, labels in patch_loader:
        print("Batch size:", images.shape, "Labels:", labels)
        
        first_image_tensor = images[0]  # shape: (C, H, W)

        first_image_pil = F.to_pil_image(first_image_tensor)

        plt.imshow(first_image_pil)
        plt.title(f"Label: {labels[0].item()}")
        plt.axis('off')
        plt.show()
        break