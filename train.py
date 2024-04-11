#!/usr/bin/python
# -*- coding:utf-8 -*-

import argparse
import os
from datetime import datetime
from utils.logger import setlogger
import logging
from utils.train_utils_base import train_utils
from transfer_learning.fine_tuning import *
from transfer_learning.dbtl import *
import torch
import warnings
print(torch.__version__)
print("CUDA :", torch.version.cuda)
warnings.filterwarnings('ignore')

args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # model and data parameters
    parser.add_argument('--model_name', type=str, default='CNN', help='the name of the model')
    parser.add_argument('--data_name', type=str, default='PHM', help='the name of the data')
    parser.add_argument('--data_dir', type=str, default='F:\Computational Engineering\MT\code\mt-transfer-learning\PHM', help='the directory of the data')

    parser.add_argument('--normlizetype', type=str, default='mean-std', help='nomalization type')
    parser.add_argument('--checkpoint_name', type=str, default='', help='the name of the checkpoint')
    parser.add_argument('--gear_health_check', type=bool, default=False, help='Enable gear health check')


    # adabn parameters
    parser.add_argument('--adabn', type=bool, default=True, help='whether using adabn')
    parser.add_argument('--eval_all', type=bool, default=False, help='whether using all samples to update the results')
    parser.add_argument('--adabn_epochs', type=int, default=3, help='the number of training process')


    # training parameters
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='the directory to save the model')
    parser.add_argument("--pretrained", type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')
    parser.add_argument('--transfer_learning', type=str, default='no_transfer_learning', help='checkpoint name')


    # optimization information
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam', help='the optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='fix', help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='5, 10', help='the learning rate decay for step and stepLR')


    # save, load and display information
    parser.add_argument('--max_epoch', type=int, default=300, help='max number of epoch')
    parser.add_argument('--print_step', type=int, default=600, help='the interval of log training information')

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()

    # sub_dir = args.model_name + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    # save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if args.transfer_learning == 'no_transfer_learning':
        if args.checkpoint_name:
            save_dir = os.path.join(args.checkpoint_dir, args.checkpoint_name)
        else:
            sub_dir = args.model_name + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
            save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    else:
        if args.checkpoint_name:
            sub_dir = 'transfer-learning-' + args.checkpoint_name
            save_dir = os.path.join(args.checkpoint_dir, sub_dir)
        else:
            sub_dir = 'transfer-learning-' + args.model_name + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
            save_dir = os.path.join(args.checkpoint_dir, sub_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, 'train.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))


    if args.transfer_learning == 'fine_tuning':
        trainer = fine_tuning(args, save_dir)
        trainer.setup()
        trainer.train()
    elif args.transfer_learning == 'dbtl':
        trainer = dbtl(args, save_dir)
        trainer.setup()
        trainer.train()
    else:
        trainer = train_utils(args, save_dir)
        trainer.setup()
        trainer.train()
