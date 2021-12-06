#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multi-view test a video classification model."""

import numpy as np
import os
import pickle
import torch
import random
import cv2
import matplotlib.pyplot as plt
import skimage.io as io

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
import slowfast.visualization.tensorboard_vis as tb
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.env import pathmgr
from slowfast.utils.meters import AVAMeter, TestMeter
import csv
from csv import reader

logger = logging.get_logger(__name__)


@torch.no_grad()
def perform_visu(test_loader, image, model, test_meter, cfg, writer=None):
    """
    For classification:
    Perform mutli-view testing that uniformly samples N clips from a video along
    its temporal axis. For each clip, it takes 3 crops to cover the spatial
    dimension, followed by averaging the softmax scores across all Nx3 views to
    form a video-level prediction. All video predictions are compared to
    ground-truth labels and the final testing performance is logged.
    For detection:
    Perform fully-convolutional testing on the full frames without crop.
    Args:
        test_loader (loader): video testing loader.
        model (model): the pretrained video model to test.
        test_meter (TestMeter): testing meters to log and ensemble the testing
            results.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter object, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable eval mode.
    model.eval()
    test_meter.iter_tic()

    for cur_iter, (inputs, labels, video_idx, meta, ctr_idx) in enumerate(test_loader):
        image_name=None
        with open(os.path.join(cfg.AVA.FRAME_LIST_DIR,'val.csv'), 'r') as read_obj:
            csv_reader = reader(read_obj)
            header = next(csv_reader)
            for row in csv_reader:
                frame_id = row[2]
                if int(frame_id) == ctr_idx[0] :
                    image_name = row[3]
        image_test = image.split('/')[8]+'/'+image.split('/')[9]
        if image_name !=None and image_name== image_test:
            actions = [0]*cfg.MODEL.NUM_CLASSES
            with open(os.path.join(cfg.DEMO.LABEL_FILE_PATH), 'r') as file:
                k=0
                for line in file:
                    if 'name' in line:
                        actions[k] = line.split('"')[1]
                        k += 1
            
            if cfg.NUM_GPUS:
                # Transfer the data to the current GPU device.
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)

                # Transfer the data to the current GPU device.
                labels = labels.cuda()
                video_idx = video_idx.cuda()
                for key, val in meta.items():
                    if isinstance(val, (list,)):
                        for i in range(len(val)):
                            val[i] = val[i].cuda(non_blocking=True)
                    else:
                        meta[key] = val.cuda(non_blocking=True)
            test_meter.data_toc()

            if cfg.DETECTION.ENABLE:
                # Compute the predictions.
                preds = model(inputs, meta["boxes"])
                ori_boxes = meta["ori_boxes"]
                metadata = meta["metadata"]
                preds = preds.detach().cpu() if cfg.NUM_GPUS else preds.detach()
                ori_boxes = (
                    ori_boxes.detach().cpu() if cfg.NUM_GPUS else ori_boxes.detach()
                )
                metadata = (
                    metadata.detach().cpu() if cfg.NUM_GPUS else metadata.detach()
                )


                if cfg.NUM_GPUS > 1:
                    preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                    ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                    metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

                test_meter.iter_toc()
                # Update and log stats.
                test_meter.update_stats(preds, ori_boxes, metadata)
                test_meter.log_iter_stats(None, cur_iter)




            test_meter.iter_tic()


            test_meter.finalize_metrics()

            num_action=[]
            for j in range(len(labels)):
                label = labels[j].cpu().numpy()
                num=0
                for l in range(len(label)):
                    if label[l]==1:
                        num+=1
                num_action.append(num)

            indices_gt= []
            for j in range(len(labels)):
                label = labels[j]
                _, ind=torch.topk(label, num_action[j])
                indices_gt.append(ind.cpu().numpy())
            
            indices_pred= []
            for j in range(len(preds)):
                pred = preds[j]
                _, ind=torch.topk(pred, num_action[j])
                indices_pred.append(ind.cpu().numpy())

            image = io.imread(image)
            bbox = ori_boxes
            imagebis= np.copy(image)
            for i in range(len(bbox)):
                imagebis= cv2.rectangle(imagebis, (int(bbox[i][1]),int(bbox[i][2])),(int(bbox[i][3]),int(bbox[i][4])),(255,i*30,i*30), 2)
                if num_action[i] !=1:
                    for j in range(num_action[i]):              
                        imagebis = cv2.putText(imagebis, '[GT] : {}'.format(actions[int(indices_gt[i][j])]), (int(bbox[i][1]), int(bbox[i][2])+(30*(j+1))), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,i*30,i*30), 2)
                        imagebis = cv2.putText(imagebis, '[{:.4f}] : {}'.format(float(preds[i][int(indices_pred[i][j])]),actions[int(indices_pred[i][j])]), (int(bbox[i][3])-250, int(bbox[i][4])-(30*(j+1))), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,i*30,i*30), 2)
                else:
                    imagebis = cv2.putText(imagebis, '[GT] : {}'.format(actions[int(indices_gt[i])]), (int(bbox[i][1]), int(bbox[i][2])+(30*(j+1))), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,i*30,i*30), 2)
                    imagebis = cv2.putText(imagebis, '[{:.4f}] : {}'.format(float(preds[i][int(indices_pred[i])]),actions[int(indices_pred[i])]), (int(bbox[i][3])-250, int(bbox[i][4])-(30*(j+1))), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,i*30,i*30), 2)
            
            io.imsave(os.path.join('method','prediction.png'),  imagebis)
            plt.figure()
            plt.axis('off')
            plt.imshow(imagebis)
            plt.show()


    return test_meter


def demo(cfg, img):
    """
    Perform multi-view testing on the pretrained video model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config.
    logger.info("Test with config:")
    logger.info(cfg)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)

    # Create video testing loaders.
    test_loader = loader.construct_loader(cfg, "test")
    logger.info("Testing model for {} iterations".format(len(test_loader)))

    if cfg.DETECTION.ENABLE:
        assert cfg.NUM_GPUS == cfg.TEST.BATCH_SIZE or cfg.NUM_GPUS == 0
        test_meter = AVAMeter(len(test_loader), cfg, mode="test")
    else:
        assert (
            test_loader.dataset.num_videos
            % (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS)
            == 0
        )
        # Create meters for multi-view testing.
        test_meter = TestMeter(
            test_loader.dataset.num_videos
            // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
            cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
            cfg.MODEL.NUM_CLASSES,
            len(test_loader),
            cfg.DATA.MULTI_LABEL,
            cfg.DATA.ENSEMBLE_METHOD,
        )

    # Set up writer for logging to Tensorboard format.
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None

    image=img

    # # Test and visualization of one image.
    test_meter = perform_visu(test_loader, image, model, test_meter, cfg, writer)
    if writer is not None:
        writer.close()
