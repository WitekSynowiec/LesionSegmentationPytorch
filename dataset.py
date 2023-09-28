import os

from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import cv2
from torchvision.io import read_image

from Nifti import Nifti


class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir,transform=None, target_transform=None):
        self.current_image_num = 0
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_labels = [image for image in os.listdir(image_dir) if image.endswith(".npy")]
        self.masks_labels = [mask for mask in os.listdir(mask_dir) if mask.endswith(".npy")]

        assert self.image_labels == self.masks_labels

        self.current_image = None, None

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, index):
        assert self.image_labels[index] == self.masks_labels[index]
        image_path = os.path.join(self.image_dir, self.image_labels[index])
        mask_path = os.path.join(self.mask_dir, self.masks_labels[index])
        image = np.load(image_path)
        mask = np.load(mask_path)

        return image, mask


if __name__ == "__main__":
    IMAGES_PATH = r'/mnt/c/Users/pegaz/Desktop/Praca-Magisterska/dataset/organized/images/'
    ANNOTATIONS_PATH = r'/mnt/c/Users/pegaz/Desktop/Praca-Magisterska/dataset/organized/annotations/'
    ORIGINAL_DATASET_PATH = r'/mnt/c/Users/pegaz/Desktop/Praca-Magisterska/dataset/zenodo_read_only'
    METADATA_FILE = r'sources.txt'
    ds = CarvanaDataset(IMAGES_PATH, ANNOTATIONS_PATH)

    image = ds.__getitem__(17000)
    print(image[0].shape)
    print(image[1].shape)

    # print(cv2.resize(ds.get_slices(500)[1], (128,128)).shape)

#%%
