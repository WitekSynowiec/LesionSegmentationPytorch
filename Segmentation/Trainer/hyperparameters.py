

import torch.cuda

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 5
IMAGES_PATH = r'/mnt/c/Users/pegaz/Desktop/Praca-Magisterska/dataset/organized/images/'
ANNOTATIONS_PATH = r'/mnt/c/Users/pegaz/Desktop/Praca-Magisterska/dataset/organized/annotations/'
PIN_MEMORY = True
LOAD_MODEL = False


