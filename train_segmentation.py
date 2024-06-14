import os
import random
import sys
from datetime import datetime

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from loss import DiceLoss, FocalDiceLoss, BCEDiceLoss
from unet import Unet

# Initialize parameters
params = {
    'exp_name': 'label_{}'.format(datetime.now().replace(microsecond=0).isoformat().replace(':', '_')),
    'dataset': '/home/zestiot/data_unet_1',
    'dloader_params': {
        'batch_size': 16,
        'num_workers': 8,
        'shuffle': True
    },
    'training_params': {
        'num_epochs': 100,
        'learning_rate': 0.001,
    },
    'loss_params': {
        'bce_weight': 0.70,
        'dice_weight': 0.30
    }
}
device = torch.device("cuda:0")

random.seed(123)
torch.manual_seed(123)
np.random.seed(123)

# Create dataloader
# 1. Define Data Iterator

def adjustImage(input_img, grayscale=False, multiple=32):
    """ Adjusts Dimensions of the Image for inference by padding with zeros along edges.

    Args:
        input_img (ndarray): Input grascale input.

    Returns:
        ndarray: Padded grayscale output.
    """
    if grayscale:
        new_shape = [None, None]
    else:
        new_shape = [None, None, 3]
    img_shape = input_img.shape
    if (input_img.shape[0] % multiple == 0): 
        new_shape[0] = input_img.shape[0]
    else: 
        new_shape[0] = img_shape[0] - (img_shape[0] % multiple) + multiple
    if (input_img.shape[1] % multiple == 0):
        new_shape[1] = input_img.shape[1]
    else:
        new_shape[1] = img_shape[1] - (img_shape[1] % multiple) + multiple

    zeros_mask = np.zeros(shape=(new_shape), dtype=input_img.dtype)
    zeros_mask[:img_shape[0], :img_shape[1]] = input_img
    return zeros_mask

class JSWDataIterator(torch.utils.data.Dataset):
    def __init__(self, phase='train'):
        self.rootdir = params['dataset']
        self.image_prefix = os.path.join(self.rootdir, 'inputs')
        self.target_prefix = os.path.join(self.rootdir, 'targets')
        self.images = os.listdir(self.image_prefix)
        random.shuffle(self.images)

        if phase == 'train':
            self.images = self.images[:int(0.75*(len(self.images)))]
        else:
            self.images = self.images[int(0.75*(len(self.images))):]

        self.inputs = [os.path.join(self.image_prefix, f)
                       for f in os.listdir(self.image_prefix)]
        self.targets = [os.path.join(self.target_prefix, f)
                        for f in os.listdir(self.target_prefix)]
        self.img_transform = A.Compose([
            #A.RandomCrop(width=352, height=352),
            #A.Rotate(),
            A.ColorJitter(brightness=[0.5,1], contrast=[0.5,1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5)
        ])
        self.img_tns_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0)
            )
        ])
        self.mask_tns_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = adjustImage(cv2.imread(self.inputs[idx]))
        mask = adjustImage(cv2.imread(self.targets[idx], 0), grayscale=True)

        # Resize images
        # image = cv2.resize(image, (1024, 768))
        # mask = cv2.resize(mask, (1024, 768))

        transformed = self.img_transform(image=image, mask=mask)
        img_tnsr = self.img_tns_transform(transformed['image'])
        mask_tnsr = self.mask_tns_transform(transformed['mask'])
        return img_tnsr, mask_tnsr


# 2. Instantiate Data Iterator and Data Loader
dataset = {phase: JSWDataIterator(phase=phase) for phase in ['train', 'val']}
dataloader = {phase: torch.utils.data.DataLoader(dataset[phase],
                                                 **params['dloader_params']) for phase in ['train', 'val']}

# Initialize model
model = Unet().to(device)
for m in model.modules():
    if isinstance(m, (torch.nn.Conv2d)):
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in')
    if isinstance(m, (torch.nn.BatchNorm2d)):
        torch.nn.init.normal_(m.weight)

# Initialize optimizer and loss function
optimizer = torch.optim.Adam(
    model.parameters(), lr=params['training_params']['learning_rate'])
criterion = {
    'train': BCEDiceLoss(params['loss_params']),
    'val': DiceLoss()
}
# Train model


def train_model(model, optimizer, criterion, dataloader, num_epochs, exp_name):
    best_loss = 100.0
    loss_graph = {'train': [], 'val': []}
    for epoch in range(num_epochs):
        print("*"*20)
        print("*   EPOCH : ", epoch + 1)
        print("*"*20)
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            epoch_loss = 0.0
            for data, label in tqdm(dataloader[phase], desc=f'{phase.upper()} progress : '):
                data = data.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(data)
                    loss = criterion[phase](output, label)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss = running_loss + loss.item() * data.size(0)
            epoch_loss = running_loss / len(dataset[phase])
            print("{0}\t EPOCH LOSS: {1:.5f}".format(
                phase.upper(), epoch_loss))
            loss_graph[phase].append(epoch_loss)

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                print("Saving Best Weights ...")
                torch.save(model.state_dict(),
                           './{}/{}_weights.pth'.format(exp_name, exp_name))
        plt.plot(loss_graph['train'], label='train')
        plt.plot(loss_graph['val'], label='valid')
        plt.xlabel('No. of Epochs')
        plt.ylabel('Loss')
        plt.title('Train loss: {:.5f} | Best Val Loss: {:.5f} | Epoch {}'.format(
            loss_graph['train'][-1], best_loss, epoch + 1))
        plt.savefig(f'./{exp_name}/{exp_name}_loss_graph.png')
        plt.draw()
        plt.pause(1e-5)
        plt.clf()
        print()


if __name__ == '__main__':
    if not os.path.isdir(params['exp_name']):
        os.mkdir(params['exp_name'])

    with open(f"./{params['exp_name']}/{params['exp_name']}.txt", 'w') as file:
        for k, v in params.items():
            if isinstance(v, dict):
                print(f'{k}:')
                file.write(f'{k}:')
                for k_, v_ in v.items():
                    print(f"\t{k_}\t: {v_}")
                    file.write(f"\t{k_}\t: {v_}")
                print()
            else:
                print(f"{k}\t: {v}")
                file.write(f"{k}\t: {v}")

    train_model(model=model, optimizer=optimizer,
                criterion=criterion, dataloader=dataloader,
                num_epochs=params['training_params']['num_epochs'],
                exp_name=params['exp_name'])
    print('\n\nDone.')
