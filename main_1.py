

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

from sklearn.metrics import accuracy_score, roc_auc_score

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models

from conv_net import *

from utils import *
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
# toDo: label smoothing
# endregion

# region plot pie plot of the classes
# plt.pie(samples_lbls.label.value_counts(), labels=[0, 1])
# endregion

# region plot some images
# fig, ax = plt.subplots(3, 3)
# image_plot_idx = np.random.choice(len(samples_lbls), 9, replace=False)
# for i_img, img in enumerate(samples_lbls.id[image_plot_idx]):
#     n, m = int(i_img / 3), i_img % 3
#
#     im = Image.open(op.join(path_train_data, img + '.tif'))
#     ax[n, m].imshow(im)
#     ax[n, m].grid(False)
#     title = 'sample no.=' + str(image_plot_idx[i_img]) + ', label=' + str(samples_lbls.label[i_img].item())
#     ax[n, m].set_title(title, fontsize=7 )
#     ax[n, m].tick_params(labelbottom=False, labelleft=False)
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
dataset_test = ImageDataset(samples_test, path_test_data, transforms_test)

# dataloader
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, drop_last=False)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=False)

n_train_loader = len(dataloader_train)
n_val_loader = len(dataloader_val)
n_test_loader = len(dataloader_test)

# test the dataloader's functionality
# train_features, train_labels = next(iter(dataloader_train))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")

# endregion

# region CNN model
# model = AlexNet(96).to(device)
# model = CNN1().to(device)
model = VGG16p2(96).to(device)
print(model)
# endregion


# region Loss and optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
# toDo: learning schedule
# endregion

# region containers for the train loop
train_auc = []
val_auc = []

# average of the auc in each epoch
train_auc_epoch = []
val_auc_epoch = []

loss_all = []
train_loss_epoch = []
val_loss_epoch = []

train_auc_epoch_2 = []  # like a cumsum
val_auc_epoch_2 = []

min_loss = np.Inf
best_model_auc = 0
# endregion

# region train loop
start_time = time.time()
for e in range(n_epochs):
    train_loss = 0
    val_loss = 0

    train_auc_e = 0
    val_auc_e = 0

    model.train()
    for i_img1, (images1, labels1) in enumerate(dataloader_train):

        lbls_gt1 = labels1.numpy()

        images1 = images1.to(device)
        labels1 = labels1.to(device)

        labels_pred1 = model(images1)
        loss = loss_func(labels_pred1, labels1)
        loss_all.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(loss.item())
        train_loss += (loss.item() / n_train_loader)
        lbl_pred1 = labels_pred1[:, -1].detach().cpu().numpy()
        auc_train = roc_auc_score(lbls_gt1, lbl_pred1)
        train_auc.append(auc_train)
        train_auc_e += auc_train / n_train_loader
        if not i_img1 % 50:
            print(f'batch{i_img1} / {n_train_loader} of train')

    model.eval()
    # with torch.no_grad():
    for i_img, (images, labels) in enumerate(dataloader_val):
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
        if not i_img % 50:
            print(f'batch{i_img} / {n_val_loader} of validation')

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
        # torch.save(model.state_dict(), 'best_model.pt')
        best_model_auc = val_auc

    print(f'Epoch {e+1} with train loss={train_loss} and validation loss={val_loss}')

time_train = time.time() - start_time
torch.save(best_model.state_dict(), './results/best_model.pt')
torch.save(model.state_dict(), './results/model.pt')
save_json('train_loss', './results', train_loss_epoch)
save_json('val_loss', './results', val_loss_epoch)
save_json('train_auc', './results', train_auc_epoch)
save_json('val_auc', './results', val_auc_epoch)
save_json('train_auc_2', './results', train_auc_epoch_2)
save_json('val_auc_2', './results', val_auc_epoch_2)
# endregion

# region train the model - toDo
# early stopping + using pre-trained GoogLENET
# leakyRelu
# endregion

# region plot the loss and AUC
plt.figure()
plt.plot(np.arange(1, n_epochs+1), train_loss_epoch, '-*', label='train loss')
plt.plot(np.arange(1, n_epochs+1), val_loss_epoch, '-*', label='validation loss')
plt.legend(loc="upper left")

plt.figure()
plt.plot(np.arange(1, n_epochs+1), train_auc_epoch, '-*',  label='train AUC')
plt.plot(np.arange(1, n_epochs+1), val_auc_epoch, '-*', label='validation AUC')
plt.legend(loc="upper left")
# endregion

# region apply the model on test data
labels_test = []
model.eval()
with torch.no_grad():
    for i_img, (images, labels) in enumerate(dataloader_test):
        images = images.to(device)
        labels_pred = best_model(images)
        lbl_pred = list(labels_pred[:, -1].detach().cpu().numpy())
        labels_test += lbl_pred

        if not i_img % 50:
            print(f'batch{i_img} / {n_test_loader} of validation')
# endregion

# region write the test predictions to the csv file
samples_test['label'] = labels_test
samples_test.to_csv('./results/submission.csv', index=False)
samples_test.info()
# endregion














