from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
import torchvision.transforms as tf
from PIL import Image
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
        mlb = MultiLabelBinarizer(classes = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16))
        labels = []
        images_box = []
        for i in range(0, len(self.list_files.get('annotations'))):
            if image_id == self.list_files.get("annotations")[i].get('image_id'):
                labels.append((self.list_files.get("annotations")[i].get('actions')))
                bbox = self.list_files.get("annotations")[i].get('bbox')
                image_box = image[int(bbox[1]):int(bbox[1])+int(bbox[3]), int(bbox[0]):int(bbox[0])+int(bbox[2])]
                images_box.append(np.array(Image.fromarray(image_box).resize((16,16))))
        images_box_tensor = torch.Tensor(images_box)
        labels = mlb.fit_transform(labels)
        labels_tensor = torch.Tensor(labels)
        n_box, n_classes = labels_tensor.size()
        if n_box>0:
            images_tensor = torch.zeros((n_box,128))
            for u in range(n_box):
                images_tensor[u] = torch.Tensor(np.resize(np.array(images_box_tensor[u].flatten()), 128))
        else:
            images_tensor = torch.Tensor([])
        return images_tensor, labels_tensor

    def __len__(self):
        return len(self.list_files.get('images'))


def train_loader(config):
    return DataLoader(dataset=Dataset(config.train_label, config.data_root), num_workers=config.num_workers,
                      batch_size=config.train_batch_size, shuffle=True)


def test_loader(config):
    return DataLoader(dataset=Dataset(config.test_label, config.data_root), num_workers=config.num_workers,
                      batch_size=config.train_batch_size, shuffle=True)