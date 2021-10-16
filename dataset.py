from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import skimage.io  as io
import scipy.io as sio
import numpy as np
import torch 
import json
import os


# VideoDataset inherit from Dataset class
class Dataset(Dataset):
    def __init__(self, label_file, data_root):
        self.list_files = json.load(open(label_file))
        self.data_root = data_root

    def __getitem__(self, index):
        image_id = self.list_files.get("images")[index].get("id")
        image_name = self.list_files.get("images")[index].get("file_name")
        image = io.imread(os.path.join(self.data_root, image_name))
        labels = []
        images_box = []
        for i in range(0, len(self.list_files.get('annotations'))):
            if image_id == self.list_files.get("annotations")[i].get('image_id'):
                #label = self.list_files.get("annotations")[i].get('actions')
                #labels.append(torch.from_numpy(np.array(self.list_files.get("annotations")[i].get('actions'))))
                labels.append(self.list_files.get("annotations")[i].get('actions'))
                bbox = self.list_files.get("annotations")[i].get('bbox')
                image_box = image[int(bbox[0]):int(bbox[0])+int(bbox[2]), int(bbox[1]):int(bbox[1])+int(bbox[3])]
                images_box.append(image_box)
                plt.figure()
                plt.imshow(image_box)
                plt.show()
                #np.concatenate(labels, np.array(label))
                #np.concatenate(images_box, image_box)
        #np.stack(images_box)
        #np.stack(labels)
        breakpoint()
        images_tensor = torch.unsqueeze(torch.Tensor(images_box), 0)
        labels_tensor = torch.unsqueeze(torch.Tensor(labels[0]), 0)
        #images_tensor = torch.from_numpy(np.array(images_box))
        #labels_tensor = torch.from_numpy(np.array(labels))
        return images_tensor, labels_tensor

    def __len__(self):
        return len(self.list_files.get('images'))


def train_loader(config):
    return DataLoader(dataset=Dataset(config.train_label, config.data_root), num_workers=config.num_workers,
                      batch_size=config.train_batch_size, shuffle=True)


def test_loader(config):
    return DataLoader(dataset=Dataset(config.test_label, config.data_root), num_workers=config.num_workers,
                      batch_size=config.train_batch_size, shuffle=True)