import sys

import torch
from torch import optim, nn
from torchvision.transforms import InterpolationMode, Compose, CenterCrop, Resize, Normalize
from tqdm import tqdm

from Segmentation.dataset import MSDataset
from Segmentation.model import UNet
from Segmentation.Trainer import hyperparameters
from Segmentation.utils import get_loaders


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(torch.device("cuda"))
        targets = targets.to(torch.device("cuda"))
        # targets = targets.float().unsqueeze(1).to(device=hyperparameters.DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def test1():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = UNet(initial_in_channels=1, initial_out_channels=1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters.LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    dataset = MSDataset(
        image_dir=hyperparameters.IMAGES_PATH,
        mask_dir=hyperparameters.ANNOTATIONS_PATH,
        transform=Compose([
            Resize(size=(128, 128), interpolation=InterpolationMode.BILINEAR)
        ]),
        target_transform=Compose([
            Resize(size=(128, 128), interpolation=InterpolationMode.NEAREST)
        ])
    )

    train_loader, val_loader = get_loaders(
        dataset=dataset,
        batch_size=hyperparameters.BATCH_SIZE
    )

    for epoch in range(hyperparameters.EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

    # train the model
    # for epoch in range(10):
    #     for data in train_loader:
    #         inputs, labels = data
    #         inputs, labels = inputs.to(device), labels.to(device)
    # optimizer.zero_grad()
    # outputs = model(inputs)
    # loss = ...
    # loss.backward()
    # optimizer.step()


def test2():
    train_loader, val_loader = get_loaders(
        hyperparameters.IMAGES_PATH,
        hyperparameters.ANNOTATIONS_PATH,
        hyperparameters.BATCH_SIZE,
        None,
        None
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(initial_in_channels=1, initial_out_channels=1).to(hyperparameters.DEVICE)

    dataset = MSDataset(
        image_dir=r'/mnt/c/Users/pegaz/Desktop/Praca-Magisterska/dataset/organized/images/',
        mask_dir=r'/mnt/c/Users/pegaz/Desktop/Praca-Magisterska/dataset/organized/annotations/',
        transform=None,
        target_transform=None
    )
    y, z = dataset.__getitem__(6)
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    print("Tensor:", x)

    # check tensor device (cpu/cuda)
    print("Tensor device:", y.device)

    # Move tensor from CPU to GPU
    # check CUDA GPU is available or not
    print("CUDA GPU:", torch.cuda.is_available())
    if torch.cuda.is_available():
        y = y.to("cuda")
        # or x=x.to("cuda")
    print(y)

    # now check the tensor device
    print("Tensor device:", y.device)


if __name__ == "__main__":
    test1()
