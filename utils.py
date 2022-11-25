
# region packages
from torch.utils.data import Dataset
from PIL import Image
import os.path as op
import os
import pandas as pd
import numpy as np
import json
# endregion


# region read the samples
def read_data(path_train_data, path_train_lbls, path_test_data):
    samples_files = os.listdir(path_train_data)
    n_sample_train = len(samples_files)
    samples_ids = [samp[:-4] for samp in samples_files]

    samples_lbls = pd.read_csv(path_train_lbls)
    lbl_samples_id = list(samples_lbls.id)
    lbl_samples = list(samples_lbls.label)
    n_lbl_samples = len(lbl_samples_id)

    assert n_lbl_samples == n_sample_train

    _, _, ind_2 = np.intersect1d(samples_ids, lbl_samples_id, return_indices=True)

    assert n_lbl_samples == len(ind_2)

    samples_test_file = os.listdir(path_test_data)
    id_test = [samp[:-4] for samp in samples_test_file]
    n_sample_test = len(id_test)
    samples_test = pd.DataFrame({'id': id_test, 'label': np.zeros(n_sample_test)})

    print('ratio of test samples to train samples=', n_sample_test / n_sample_train)
    return samples_lbls, samples_test
# endregion


# region class Dataset
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class ImageDataset(Dataset):
    def __init__(self, samples, img_dir, transform=None):
        super().__init__()
        self.image_labels = samples.values
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_path = op.join(self.img_dir, self.image_labels[idx, 0] + '.tif')
        image = Image.open(img_path)
        label = self.image_labels[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

# endregion

# region load and save
def save_json_from_numpy(filename, folder, var):
    # save numpy array as json file

    with open(op.join(folder, filename), "w") as f:
        json.dump(var.tolist(), f)


def load_json_to_numpy(filename, folder):
    # load json file to a numpy array

    with open(op.join(folder, filename), "r") as f:
        saved_data = json.load(f)

    var = np.array(saved_data)
    return var


def save_json(filename, folder, var):
    # save list to json

    with open(op.join(folder,filename), "w") as f:
        json.dump(var, f)


def load_json(filename,folder):
    # load list from json

    with open(op.join(folder,filename), "r") as f:
        saved_data = json.load(f)

    var = saved_data
    return var


def save_pickle(file, folder, var):
    import pickle
    with open(op.join(folder, file), "wb") as output_file:
        pickle.dump(var, output_file)


def load_pickle(file,folder):
    import pickle
    with open(op.join(folder, file), "rb") as input_file:
        var = pickle.load(input_file)
    return var

# endregion

