from dataset import train_loader, test_loader
import argparse
import torch
import time
import copy
import torch.nn as nn
import torch.optim as optim
from model.Models import SDConv
from model.Optim import ScheduledOptim
import numpy as np
from eval import visualize_pred, edit_score, f_score
from tqdm import tqdm
import sklearn.metrics as metrics


def test(model, test_loader, device, criterion, n_class, dataset):
    predicted_result = []

    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    running_mAP = 0.0

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            _, n_box, n_classes = labels.size()
            sup=0
            if n_box != 0:
                _, input_len, _ = inputs.size()
                idx = torch.arange(1, input_len + 1)
                attn = torch.arange(0, input_len)
                idx = idx.unsqueeze(0).to(device)
                attn = attn.unsqueeze(0).to(device)
                for j in range(16):
                    #Adapt labels
                    label_box=[]
                    for k in range (n_box):
                        if labels[0][k][j] == 0: #0 because batch_size=1
                            label_box.append(0) 
                        else:
                            label_box.append(j)
                    
                    outputs = model(inputs, idx, attn)
                    outputs = outputs.squeeze(0)
                    labels_fin = torch.Tensor(label_box).long()
                    labels_fin =labels_fin.to(device)
                    _, predicted = torch.max(outputs, 1)
                    loss = criterion(outputs, labels_fin)
                    running_loss += loss.item()
                    running_corrects = (predicted == labels_fin).sum()
                    running_acc += running_corrects.double() / labels_fin.size(0)
                    predicted_result.append(predicted)

                for u in range(n_box):
                    pred = np.zeros(16)
                    for l in range(16):
                        if l in predicted_result[u]:
                            pred[l] = 1
                    running_mAP += metrics.average_precision_score(labels[0][u].cpu().numpy(), pred, average='macro')

            else:
                sup = sup + 1
            
            

    
    test_mAP = running_mAP / (len(test_loader)-sup)
    #test_loss = running_loss / ((len(test_loader)*16)-sup)
    test_accuracy = running_acc / ((len(test_loader)*16)-sup)


    print('Test Accuray: {:.4f}'.format(test_accuracy))
    #print('Test Loss: {:.4f}'.format(test_loss))
    print('Test mAP: {:.4f}'.format(test_mAP))




def main():
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
    parser.add_argument('--test_label', type=str, default='test_fold2.json', help='test label path.')

    config = parser.parse_args()
    print(config)

    # ========= Loading Dataset =========#
    trainX_loader = train_loader(config)
    testX_loader = test_loader(config)

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

    criterion = nn.CrossEntropyLoss()
    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, sdConv.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        config.d_model, config.n_warmup_steps)

    sdConv.load_state_dict(torch.load('baseline.pth'))

    test(sdConv, testX_loader, device, criterion, config.n_classes, config.dataset)



if __name__ == '__main__':
    main()
