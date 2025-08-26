import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class AffectNetMultiLabelDataset(Dataset):
    def __init__(self, image_folder, annotation_folder, transform=None):
        self.image_folder = image_folder
        self.annotation_folder = annotation_folder
        self.transform = transform

        self.ids = []
        for fname in os.listdir(annotation_folder):
            if fname.endswith('_exp.npy'):
                base_id = fname.split('_')[0]
                img_path = os.path.join(image_folder, base_id + '.jpg')
                aro_path = os.path.join(annotation_folder, base_id + '_aro.npy')
                val_path = os.path.join(annotation_folder, base_id + '_val.npy')
                ind_path = os.path.join(annotation_folder, base_id + '_lnd.npy')

                if os.path.exists(img_path) and \
                   os.path.exists(aro_path) and \
                   os.path.exists(val_path) and \
                   os.path.exists(ind_path):
                    self.ids.append(base_id)

        print(f"Found {len(self.ids)} samples with valid image and all 4 annotations.")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.image_folder, f'{img_id}.jpg')
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        labels = {
            'exp': torch.tensor(int(np.load(os.path.join(self.annotation_folder, f'{img_id}_exp.npy'))), dtype=torch.long),
            'aro': torch.tensor(float(np.load(os.path.join(self.annotation_folder, f'{img_id}_aro.npy'))), dtype=torch.float32),
            'val': torch.tensor(float(np.load(os.path.join(self.annotation_folder, f'{img_id}_val.npy'))), dtype=torch.float32),
            'lnd': torch.tensor(np.load(os.path.join(self.annotation_folder, f'{img_id}_lnd.npy')), dtype=torch.float32)
        }

        return image, labels


# Default transform
default_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
