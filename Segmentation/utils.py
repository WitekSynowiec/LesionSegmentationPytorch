from torch.utils.data import random_split, DataLoader
from torchvision.transforms import Resize, InterpolationMode, Compose
from Segmentation.dataset import MSDataset

"""
Function returns dataloader for given dataset.
"""


def get_loaders(
        dataset,
        batch_size,
        split_ratio=(0.8, 0.2),
        num_workers=4,
        pin_memory=True
):
    train_dataset, val_dataset = random_split(dataset=dataset, lengths=list(split_ratio))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_loader, val_loader
