import torch
import os

from torchvision import datasets
from torchvision import transforms


TRAIN = 'train'
VAL = 'val'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATA_TRANSFORMS_MAP = {
    TRAIN: transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.226, 0.224, 0.225])
    ]),
    VAL: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.226, 0.224, 0.225])
    ])
}
DATA_DIR = f'{os.getcwd()}/src/transfer_learning/data/hymenoptera_data'
IMAGE_DATASETS = {
        x: datasets.ImageFolder(os.path.join(DATA_DIR, x), DATA_TRANSFORMS_MAP[x]) for x in (DATA_TRANSFORMS_MAP.keys())
    }
DATASET_SIZE = {x: len(IMAGE_DATASETS[x]) for x in DATA_TRANSFORMS_MAP.keys()}
CLASS_NAMES = IMAGE_DATASETS[TRAIN].classes