

# region packages
import os
import os.path as op
import copy
import time

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook as tqdm
from PIL import Image

from sklearn.metrics import accuracy_score,roc_auc_score

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from conv_net import CNN

from utils import read_data, ImageDataset
# endregion

print(mpl.rcParams['backend'])
device = torch.device('mps')

# region settings
batch_size = 128
learning_rate = 0.001
n_epochs = 10

# endregion

# region path definition
path_base = '/Users/mina/Documents/Academics/Data/Kaggle/histopathologic-cancer-detection/'
path_train_data = op.join(path_base, 'train')
path_test_data = op.join(path_base, 'test')
path_train_lbls = op.join(path_base, 'train_labels.csv')
# endregion

# region collect sample names
samples_lbls, samples_test = read_data(path_train_data, path_train_lbls, path_test_data)
# endregion

# region plot pie plot of the classes
plt.pie(samples_lbls.label.value_counts(), labels=[0, 1])
# endregion

# region plot some images
fig, ax = plt.subplots(3, 3)
image_plot_idx = np.random.choice(len(samples_lbls), 9, replace=False)
for i_img, img in enumerate(samples_lbls.id[image_plot_idx]):
    n, m = int(i_img / 3), i_img % 3

    im = Image.open(op.join(path_train_data, img + '.tif'))
    ax[n, m].imshow(im)
    ax[n, m].grid(False)
    title = 'sample no.=' + str(image_plot_idx[i_img]) + ', label=' + str(samples_lbls.label[i_img].item())
    ax[n, m].set_title(title, fontsize=7 )
    ax[n, m].tick_params(labelbottom=False, labelleft=False)
# endregion

# region validation set
samples_train, samples_val = train_test_split(samples_lbls, stratify=samples_lbls.label, test_size=0.1)
print('train sample size=', len(samples_train), '- test sample size=', len(samples_val))
# endregion


# region define transforms
# https://pytorch.org/vision/stable/transforms.html
transforms_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                      transforms.RandomRotation(45), transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transforms_test = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transforms_val = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# endregion


# region datasets and dataloaders
# define the datasets of the train, test, val
dataset_train = ImageDataset(samples_train, path_train_data, transforms_train)
dataset_val = ImageDataset(samples_val, path_train_data, transforms_val)
dataset_test = ImageDataset(samples_test, path_train_data, transforms_test)

# dataloader
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

n_train_loader = len(dataloader_train)
n_val_loader = len(dataloader_val)
n_test_loader = len(dataloader_test)

# test the dataloader's functionality
# train_features, train_labels = next(iter(dataloader_train))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")

# endregion

# region CNN model
model = CNN().to(device)
print(model)
# end region


# region Loss and optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
# endregion


train_auc = []
val_auc = []

# average of the auc in each epoch
train_auc_epoch = []
val_auc_epoch = []

train_loss_epoch = []
val_loss_epoch = []

train_auc_epoch_2 = []  # like a cumsum
val_auc_epoch_2 = []

min_loss = np.Inf
best_model_auc = 0

# region train loop
start_time = time.time()
for e in range(n_epochs):
    train_loss = 0
    val_loss = 0

    train_auc_e = 0
    val_auc_e = 0

    model.train()
    for i_img, (images, labels) in enumerate(dataloader_train):
        lbls_gt = labels.numpy()

        images = images.to(device)
        labels = labels.to(device)

        labels_pred = model(images)
        loss = loss_func(labels_pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() / n_train_loader
        lbl_pred = labels_pred[:, -1].detach().cpu().numpy()
        auc_train = roc_auc_score(lbls_gt, lbl_pred)
        train_auc.append(auc_train)
        train_auc_e += auc_train / n_train_loader

    model.eval()
    for i_img, (images , labels) in enumerate(dataloader_val):
        lbls_gt = labels.numpy()

        images = images.to(device)
        labels = labels.to(device)

        labels_pred = model(images)
        loss = loss_func(labels_pred, labels)
        val_loss += loss.item() / n_val_loader
        lbl_pred = labels_pred[:, -1].detach().cpu().numpy()
        auc_val = roc_auc_score(lbls_gt, lbl_pred)
        val_auc.append(auc_val)
        val_auc_e += auc_val / n_val_loader

    # save the epoch metrics
    train_loss_epoch.append(train_loss)
    val_loss_epoch.append(val_loss)

    train_auc_epoch.append(train_auc_e)
    val_auc_epoch.append(val_auc_e)

    train_auc_epoch_2.append(np.mean(train_auc))
    val_auc_epoch_2.append(np.mean(val_auc))

    # save the best model
    if val_loss <= min_loss:
        best_model = copy.deepcopy(model)
        min_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
        best_model_auc = val_auc

    print(f'Epoch {e+1} with train loss={train_loss} and validation loss={val_loss}')

# endregion

# region train the model - toDo
# early stopping + using pre-trained GoogLENET
# leakyRelu
# endregion















