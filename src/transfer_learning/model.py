import os
import time
from tempfile import TemporaryDirectory

import torch
from PIL import Image
from matplotlib import pyplot as plt

from utils import imshow
from constants import TRAIN, VAL, DEVICE, CLASS_NAMES, DATASET_SIZE


def train_model(model, criterion, optimizer, scheduler, dataloaders, num_epochs=25):
    since = time.time()

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)
            for phase in [TRAIN, VAL]:
                # Change model mode accordingly
                if phase == TRAIN:
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(DEVICE)
                    labels = labels.to(DEVICE)

                    optimizer.zero_grad()

                    # Forward, track history for training
                    with torch.set_grad_enabled(phase == TRAIN):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        if phase == TRAIN:
                            loss.backward()
                            optimizer.step()
                    # Stats
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == TRAIN:
                    scheduler.step()
                epoch_loss = running_loss / DATASET_SIZE[phase]
                epoch_acc = running_corrects.double() / DATASET_SIZE[phase]
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == VAL and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)
            print()
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        model.load_state_dict(torch.load(best_model_params_path))
    return model


def visualize_model_predictions(model, data_transforms, img_path):
    was_training = model.training
    model.eval()

    img = Image.open(img_path)
    img = data_transforms[VAL](img)
    img = img.unsqueeze(0)
    img = img.to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

        ax = plt.subplot(2,2,1)
        ax.axis('off')
        ax.set_title(f'Predicted: {CLASS_NAMES[preds[0]]}')
        imshow(img.cpu().data[0])

        model.train(mode=was_training)


def visualize_model(model, dataloaders, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders[VAL]):
            inputs = inputs.to(DEVICE)
            labels = inputs.to(DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {CLASS_NAMES[preds[j]]}')
                imshow(inputs.cpu().data[j])
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
