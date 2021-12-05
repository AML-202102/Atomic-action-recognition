#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args

from .demo_psi import demo
from .test_net import test
from .train_net import train


def main(cfg, opt):
    """
    Main function to spawn the train and test process.
    """    
    # Perform training.
    cfg.TRAIN.ENABLE = opt.train
    cfg.TEST.ENABLE = opt.test
    cfg.DEMO.ENABLE = opt.demo

    if cfg.TRAIN.ENABLE:
        launch_job(cfg=cfg, init_method=opt.init_method, func=train)

    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=opt.init_method, func=test)

    # Run demo.
    if cfg.DEMO.ENABLE:
        demo(cfg, opt.img)


if __name__ == "__main__":

    main(cfg, opt)
