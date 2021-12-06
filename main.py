from baseline.PSIAVA import train as base_PSI
from baseline.JIGSAWS import train as base_JIGSAWS
from method.tools import run_net
from method.slowfast.config.defaults import assert_and_infer_cfg
from method.slowfast.utils.misc import launch_job
from method.slowfast.utils.parser import load_config, parse_args

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
        self.parser.add_argument('--save_path', default="/media/user_home0/mverlyck/AMLProject/PSI-AVA_code/SdConv/split8", help='save path')

        self.parser.add_argument('--epoch', type=int, default=30)
        self.parser.add_argument('--train_batch_size', type=int, default=1, help='training batch size')
        self.parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')

        self.parser.add_argument('--cuda', default=True, help='use cuda?')
        self.parser.add_argument('--lr', type=float, default=0.01, help='learning rate. Default=0.01')

        self.parser.add_argument('--d_model', type=int, default=128, help='model dimension')
        self.parser.add_argument('--d_inner_hid', type=int, default=512, help='hidden_state dim')
        self.parser.add_argument('--d_k', type=int, default=16, help='key')
        self.parser.add_argument('--d_v', type=int, default=16, help='value')
        self.parser.add_argument('--n_classes', type=int, default=16, help='no.of surgical gestures')
        self.parser.add_argument('--dropout', type=float, default=0.5)
        self.parser.add_argument('--n_position', type=int, default=5000, help='max sequence len')

        # parameters for the dilated conv net
        self.parser.add_argument('--n_dlayers', type=int, default=10, help='no.of dilated layers')
        self.parser.add_argument('--num_f_maps', type=int, default=128, help='dilated layer output')

        self.parser.add_argument('--n_head', type=int, default=1, help='no.of attention head')
        self.parser.add_argument('--n_layers', type=int, default=1, help='no.of encoder layers')
        self.parser.add_argument('--n_warmup_steps', type=int, default=4000, help='optimization')

        self.parser.add_argument('--num_workers', type=int, default=0, help='number of workers.')
        self.parser.add_argument('--data_root', type=str, default='/media/user_home0/mverlyck/AMLProject/PSI-AVA_code/PSI-AVA/STFeatures', help='data root path.')
        self.parser.add_argument('--train_label', type=str, default='/media/user_home0/mverlyck/AMLProject/PSI-AVA_code/PSI-AVA/splits/Split_8/train.txt', help='train label path.')
        self.parser.add_argument('--test_label', type=str, default='/media/user_home0/mverlyck/AMLProject/PSI-AVA_code/PSI-AVA/splits/Split_8/train.txt', help='test label path.')

        self.parser.add_argument('--train', action='store_true', default=False)
        self.parser.add_argument('--test', action='store_true', default=False)
        self.parser.add_argument('--checkpoint', default='/media/user_home0/mverlyck/AMLProject/PSI-AVA_code/SdConv/split8.pth')

        self.parser.add_argument('--method', type=str, default = 'MVIT', help='method to use')

        self.parser.add_argument('--img', type=str, default='/media/user_home0/mverlyck/AMLProject/PSI-AVA_code/PSI-AVA/data/CASE001/00035.jpg', help='path to image for demo')
        self.parser.add_argument('--demo', action='store_true', default=False)

        self.parser.add_argument(
            "--shard_id",
            help="The shard id of current node, Starts from 0 to num_shards - 1",
            default=0,
            type=int,
        )
        self.parser.add_argument(
            "--num_shards",
            help="Number of shards using by the job",
            default=1,
            type=int,
        )
        self.parser.add_argument(
            "--init_method",
            help="Initialization method, includes TCP or shared file-system",
            default="tcp://localhost:9999",
            type=str,
        )
        self.parser.add_argument(
            "--cfg",
            dest="cfg_file",
            help="Path to the config file",
            default="method/configs/PSI-AVA/MVIT.yaml",
            type=str,
        )
        self.parser.add_argument(
            "opts",
            help="See slowfast/config/defaults.py for all options",
            default=None,
            nargs=argparse.REMAINDER,
        )

 

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)
    
        return opt
        


def main(opt):
    

    if opt.method == 'baseline_PSIAVA':
        base_PSI.main(opt)

    elif opt.method == 'baseline_JIGSAWS':
        base_JIGSAWS.main(opt)

    elif opt.method == 'MVIT':
        cfg = load_config(opt)
        cfg = assert_and_infer_cfg(cfg)
        run_net.main(cfg, opt)




        
       
if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
