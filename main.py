from baseline.PSIAVA import train as base_PSI
from baseline.JIGSAWS import train as base_JIGSAWS

import argparse
import torch
import time
import copy
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sklearn.metrics as metrics



class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Baseline parser')
        self.parser.add_argument('--save_path', default="/media/user_home0/mverlyck/AMLProject/JIGSAWS_code/SdConv/split8", help='save path')

        self.parser.add_argument('--epoch', type=int, default=30)
        self.parser.add_argument('--train_batch_size', type=int, default=1, help='training batch size')
        self.parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')

        self.parser.add_argument('--cuda', default=True, help='use cuda?')
        self.parser.add_argument('--lr', type=float, default=0.01, help='learning rate. Default=0.01')

        self.parser.add_argument('--d_model', type=int, default=128, help='model dimension')
        self.parser.add_argument('--d_inner_hid', type=int, default=512, help='hidden_state dim')
        self.parser.add_argument('--d_k', type=int, default=16, help='key')
        self.parser.add_argument('--d_v', type=int, default=16, help='value')
        self.parser.add_argument('--n_classes', type=int, default=15, help='no.of surgical gestures')
        self.parser.add_argument('--dropout', type=float, default=0.5)
        self.parser.add_argument('--n_position', type=int, default=5000, help='max sequence len')

        # parameters for the dilated conv net
        self.parser.add_argument('--n_dlayers', type=int, default=10, help='no.of dilated layers')
        self.parser.add_argument('--num_f_maps', type=int, default=128, help='dilated layer output')

        self.parser.add_argument('--n_head', type=int, default=1, help='no.of attention head')
        self.parser.add_argument('--n_layers', type=int, default=1, help='no.of encoder layers')
        self.parser.add_argument('--n_warmup_steps', type=int, default=4000, help='optimization')

        self.parser.add_argument('--num_workers', type=int, default=0, help='number of workers.')
        self.parser.add_argument('--data_root', type=str, default='/media/user_home0/mverlyck/AMLProject/JIGSAWS/STFeatures', help='data root path.')
        self.parser.add_argument('--train_label', type=str, default='/media/user_home0/mverlyck/AMLProject/JIGSAWS/splits/Split_8/train.txt', help='train label path.')
        self.parser.add_argument('--test_label', type=str, default='/media/user_home0/mverlyck/AMLProject/JIGSAWS/splits/Split_8/train.txt', help='test label path.')

        self.parser.add_argument('--train', action='store_true', default=False)
        self.parser.add_argument('--test', action='store_true', default=False)
        self.parser.add_argument('--checkpoint', default='/media/user_home0/mverlyck/AMLProject/PSI-AVA_code/SdConv/split8.pth')

        self.parser.add_argument('--method', type=str, default = 'MVIT', help='method to use')

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)
    
        return opt
        
    def init(self, args=''):
        opt = self.parse(args)
        return opt


def main(opt):
    

    if opt.method == 'baseline_PSIAVA':
        base_PSI.main(opt)

    elif opt.method == 'baseline_JIGSAWS':
        base_JIGSAWS.main(opt)
    # parser = argparse.ArgumentParser(description='video parsing')

    # parser.add_argument('--image', type='str', default=False, help='path to image for demo')

    # parser.add_argument('--demo', action='store_true', default=False)
    # parser.add_argument('--test', action='store_true', default=False)
    # parser.add_argument('--checkpoint', default='/media/user_home0/mverlyck/AMLProject/JIGSAWS_code/SdConv/split8.pth')

    # config = parser.parse_args()
    # print(config) 
        
       

if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
