
from dataset1 import train_loader, test_loader
import argparse
import torch
import time
import copy
import torch.nn as nn
import torch.optim as optim
from model.Models import SDConv
from model.Optim import ScheduledOptim
import numpy as np
from tqdm import tqdm
import sklearn.metrics as metrics
import random
import json
import skimage.io  as io
import os
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description='Arguments')
parser.add_argument('--dataset', default='data', help='data path')

parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--train_batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')

parser.add_argument('--cuda', default=True, help='use cuda?')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate. Default=0.01')

parser.add_argument('--d_model', type=int, default=128, help='model dimension')
parser.add_argument('--d_inner_hid', type=int, default=512, help='hidden_state dim')
parser.add_argument('--d_k', type=int, default=16, help='key')
parser.add_argument('--d_v', type=int, default=16, help='value')
parser.add_argument('--n_classes', type=int, default=16, help='no.of surgical gestures')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--n_position', type=int, default=5000, help='max sequence len')

# parameters for the dilated conv net
parser.add_argument('--n_dlayers', type=int, default=10, help='no.of dilated layers')
parser.add_argument('--num_f_maps', type=int, default=128, help='dilated layer output')

parser.add_argument('--n_head', type=int, default=1, help='no.of attention head')
parser.add_argument('--n_layers', type=int, default=1, help='no.of encoder layers')
parser.add_argument('--n_warmup_steps', type=int, default=4000, help='optimization')

parser.add_argument('--num_workers', type=int, default=0, help='number of workers.')
parser.add_argument('--data_root', type=str, default='data', help='data root path.')
parser.add_argument('--train_label', type=str, default='train_fold1.json', help='train label path.')
parser.add_argument('--test_label', type=str, default='test_fold1.json', help='test label path.')

config = parser.parse_args()
print(config)

# Using Cuda
device = torch.device("cuda" if config.cuda else "cpu")
if config.cuda and not torch.cuda.is_available():
        raise Exception('No GPU found, please run without --cuda')
print(device)
# ========= Preparing SdConv =========#
sdConv = SDConv(
        n_position=config.n_position,
        n_classes=config.n_classes,
        n_dlayers=config.n_dlayers,
        num_f_maps=config.num_f_maps,
        d_model=config.d_model,
        d_inner=config.d_inner_hid,
        n_layers=config.n_layers,
        n_head=config.n_head,
        d_k=config.d_k,
        d_v=config.d_v,
        dropout=config.dropout
    ).to(device)

sdConv.load_state_dict(torch.load('baseline.pth'))

model=sdConv
list_files = json.load(open(config.test_label))
n = len(list_files.get('images'))
index = int(random.choice(np.linspace(0,n,n+1)))

image_name = list_files.get('images')[index].get('file_name')

image_id = list_files.get("images")[index].get("id")
labels = []
images_box = []
image_visu = []
mlb = MultiLabelBinarizer(classes = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16))
image = io.imread(os.path.join(config.data_root, image_name))
for i in range(0, len(list_files.get('annotations'))):
    if image_id == list_files.get("annotations")[i].get('image_id'):
        labels.append((list_files.get("annotations")[i].get('actions')))
        bbox = list_files.get("annotations")[i].get('bbox')
        image_box = image[int(bbox[1]):int(bbox[1])+int(bbox[3]), int(bbox[0]):int(bbox[0])+int(bbox[2])]
        image_visu.append(image_box)
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

if n_box == 0:
	print('Image does not have bounding boxes')
else:
    sdConv.eval()
    inputs, labels = images_tensor.to(device), labels_tensor.to(device)
    input_len, _ = inputs.size()
    idx = torch.arange(1, input_len + 1)
    attn = torch.arange(0, input_len)
    idx = idx.unsqueeze(0).to(device)
    attn = attn.unsqueeze(0).to(device)
    predicted_result=[]
    labels_fin_tot=[]
    for j in range(16):
        #Adapt labels
        label_box=[]
        for k in range (n_box):
            if labels[k][j] == 0: 
                label_box.append(0) 
            else:
                label_box.append(j)
        # forward
        outputs = model(inputs.unsqueeze(0), idx, attn)
        outputs = outputs.squeeze(0)
        labels_fin = torch.Tensor(label_box).long()
        labels_fin =labels_fin.to(device)
        _, predicted = torch.max(outputs, 1)
        predicted_result.append(predicted)
    
    pred_tot=[]
    for u in range(n_box):
        pred = np.zeros(16)
        for l in range(16):
            if l in predicted_result[u]:
                pred[l] = 1
        pred_tot.append(pred)

    plt.figure('Baseline visualization')
    for i in range(n_box):
        plt.subplot(1,n_box,i+1)
        plt.axis('off')
        plt.imshow(image_visu[i])
        pred=[]
        gt=[]
        for d in range(16):
            if pred_tot[i][d] != 0:
                pred.append(d)
            if labels[i][d] != 0:
                gt.append(d)
        actions = list_files.get('action_categories')
        annot_gt=[]
        annot_pred=[]
        for o in range(len(actions)):
            if actions[o].get('id') in gt:
                annot_gt.append(actions[o].get('name'))
            if actions[o].get('id') in pred:
                annot_pred.append(actions[o].get('name'))
        gt_txt = ''
        pred_txt = ''
        for f in range(len(annot_gt)):
            gt_txt = gt_txt + ' ' + str(annot_gt[f])
        for q in range(len(annot_pred)):
            pred_txt = pred_txt + ' ' + str(annot_pred[q])
        if len(pred_txt)==0:
            pred_txt = 'None'
        plt.title('Groundtruth: {} \n Prediction: {}'.format(gt_txt,pred_txt))
    plt.show()
