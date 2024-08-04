from model import train_model, visualize_model_predictions, visualize_model
from constants import DEVICE, TRAIN, CLASS_NAMES, IMAGE_DATASETS, DATA_TRANSFORMS_MAP
from utils import imshow
import torch
import os
import torch.nn as nn
import torch.optim as optim

from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import models
import matplotlib.pyplot as plt


NUM_EPOCHS = int(os.getenv('EPOCHS', 2))

cudnn.benchmark = True
plt.ion()
dataloaders = {
    x: torch.utils.data.DataLoader(IMAGE_DATASETS[x], batch_size=4, shuffle=True, num_workers=0) for x in
    DATA_TRANSFORMS_MAP.keys()
}

inputs, classes = next(iter(dataloaders[TRAIN]))
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[CLASS_NAMES[x] for x in classes])

# Fine tuning
model_ft = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))

model_ft = model_ft.to(DEVICE)
criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, num_epochs=NUM_EPOCHS)
visualize_model(model_ft, dataloaders)

model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')
for param in model_conv.parameters():
    param.requires_grad = False

num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(DEVICE)
criterion = nn.CrossEntropyLoss()

optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# decay LR by a factor of .1 every 7th epoch
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, dataloaders, num_epochs=NUM_EPOCHS)

visualize_model(model_conv, dataloaders)
plt.ioff()
plt.show()

visualize_model_predictions(
    model_conv,
    DATA_TRANSFORMS_MAP,
    img_path=f'{os.getcwd()}/src/transfer_learning/data/hymenoptera_data/val/bees/72100438_73de9f17af.jpg'
)

plt.ioff()
plt.show()
