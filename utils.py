
# region packages
from torch.utils.data import Dataset
from PIL import Image
import os.path as op
import os
import pandas as pd
import numpy as np
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

