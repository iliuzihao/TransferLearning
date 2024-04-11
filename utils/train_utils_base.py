#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging
import os
import time
import warnings

import torch
from torch import nn
from torch import optim
import numpy as np

import models, datasets
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from torch.utils.tensorboard import SummaryWriter

def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.eval()
class train_utils:
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir
        self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, 'tensorboard_logs'))


    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer
        :param args:
        :return:
        """
        args = self.args

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        # Load the datasets
        Dataset = getattr(datasets, args.data_name)
        dataset = Dataset(data_dir=args.data_dir,
                          gear_health_check=args.gear_health_check, normlizetype=args.normlizetype)
        print(dataset.num_classes)
        self.datasets = {}


        # if isinstance(args.transfer_task[0], str):
        #    #print( args.transfer_task)
        #    args.transfer_task = eval("".join(args.transfer_task))


        self.datasets['source_train'], self.datasets['source_val'], self.datasets['target_val'] = dataset.data_split(args.gear_health_check, transfer_learning=False)
        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x.split('_')[1] == 'train' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False))
                            for x in ['source_train', 'source_val', 'target_val']}

        # Define the model


        self.model = getattr(models, args.model_name)(args.pretrained)
        if(args.model_name == 'MLP'):
            # self.model.classifier = nn.Linear(10, dataset.num_classes)
            self.model.fc = nn.Linear(self.model.classifier.in_features, dataset.num_classes)
        else:
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, dataset.num_classes)


        if args.adabn:
            self.model_eval = getattr(models, args.model_name)(args.pretrained)
            if (args.model_name == 'MLP'):
                # self.model.classifier = nn.Linear(10, dataset.num_classes)
                self.model_eval.fc = nn.Linear(self.model_eval.classifier.in_features, dataset.num_classes)
            else:
                self.model_eval.fc = torch.nn.Linear(self.model_eval.fc.in_features, dataset.num_classes)

        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)
            if args.adabn:
                self.model_eval = torch.nn.DataParallel(self.model_eval)


        # Define the optimizer
        if args.opt == 'sgd':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                        weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")

        # Define the learning rate decay
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'cos':
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 20, 0)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")

        self.start_epoch = 0

        # Invert the model and define the loss
        self.model.to(self.device)
        if args.adabn:
            self.model_eval.to(self.device)
        self.criterion = nn.CrossEntropyLoss()


    def train(self):
        """
        Training process
        :return:
        """
        args = self.args

        step = 0
        best_acc = 0.0
        batch_count = 0
        batch_loss = 0.0
        batch_acc = 0
        step_start = time.time()


        for epoch in range(self.start_epoch, args.max_epoch):

            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            # Update the learning rate
            if self.lr_scheduler is not None:
                # self.lr_scheduler.step(epoch)
                logging.info('current lr: {}'.format(self.lr_scheduler.get_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))


            # Each epoch has a training and val phase
            for phase in ['source_train', 'source_val', 'target_val']:  #add 'target_val' later
                # Define the temp variable
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0

                all_labels = []
                all_predictions = []

                # Set model to train mode or test mode
                if phase != 'target_val':
                    if phase=='source_train':
                       self.model.train()
                    if phase=='source_val':
                       self.model.eval()
                else:
                    if args.adabn:
                        torch.save(self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict(),
                                   os.path.join(self.save_dir, 'model_temp.pth'))
                        # self.model_eval.load_state_dict(torch.load(os.path.join(self.save_dir, 'model_temp.pth')))
                        #
                        # state_dict = torch.load(os.path.join(self.save_dir, 'model_temp.pth'))
                        # new_state_dict = {k: v for k, v in state_dict.items() if 'classifier' not in k}
                        # self.model_eval.load_state_dict(new_state_dict, strict=False)
                        state_dict = torch.load(os.path.join(self.save_dir, 'model_temp.pth'))
                        # 仅加载匹配的键，忽略不匹配的键
                        self.model_eval.load_state_dict(state_dict, strict=False)

                        self.model_eval.train()
                        self.model_eval.apply(apply_dropout)
                        with torch.set_grad_enabled(False):

                            for i in range(args.adabn_epochs):
                                if args.eval_all:
                                    for batch_idx, (inputs, _) in enumerate(self.dataloaders['target_val']):
                                        if batch_idx == 0:
                                            inputs_all = inputs
                                        else:
                                            inputs_all = torch.cat((inputs_all, inputs), dim=0)
                                    inputs_all = inputs_all.to(self.device)
                                    _ = self.model_eval(inputs_all)
                                else:
                                    for i in range(args.adabn_epochs):
                                        for batch_idx, (inputs, _) in enumerate(self.dataloaders['target_val']):
                                            inputs = inputs.to(self.device)
                                            _ = self.model_eval(inputs)
                        self.model_eval.eval()
                    else:
                        self.model.eval()




                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # Do the learning process, in val, we do not care about the gradient for relaxing
                    with torch.set_grad_enabled(phase == 'source_train'):
                        # forward
                        if args.adabn:
                            if phase != 'target_val':
                                logits = self.model(inputs)
                            else:
                                logits = self.model_eval(inputs)
                        else:
                            logits = self.model(inputs)
                        loss = self.criterion(logits, labels)
                        pred = logits.argmax(dim=1)
                        correct = torch.eq(pred, labels).float().sum().item()
                        loss_temp = loss.item() * inputs.size(0)
                        epoch_loss += loss_temp
                        epoch_acc += correct

                        # Calculate the training information
                        if phase == 'source_train':
                            # backward
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            batch_loss += loss_temp
                            batch_acc += correct
                            batch_count += inputs.size(0)
                            # Print the training information
                            if step % args.print_step == 0:
                                batch_loss = batch_loss / batch_count
                                batch_acc = batch_acc / batch_count
                                temp_time = time.time()
                                train_time = temp_time - step_start
                                step_start = temp_time
                                batch_time = train_time / args.print_step if step != 0 else train_time
                                sample_per_sec = 1.0*batch_count/train_time
                                logging.info('Epoch: {} [{}/{}], Train Loss: {:.4f} Train Acc: {:.4f},'
                                             '{:.1f} examples/sec {:.2f} sec/batch'.format(
                                    epoch, batch_idx*len(inputs), len(self.dataloaders[phase].dataset),
                                    batch_loss, batch_acc, sample_per_sec, batch_time
                                ))
                                batch_acc = 0
                                batch_loss = 0.0
                                batch_count = 0
                            step += 1

                            self.writer.add_scalar('Loss/train', batch_loss, global_step=step)

                            self.writer.add_scalar('Accuracy/train', batch_acc, global_step=step)

                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(torch.softmax(logits, dim=1).cpu().detach().numpy())
                # Print the train and val information via each epoch
                epoch_loss = epoch_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = epoch_acc / len(self.dataloaders[phase].dataset)
                logging.info('Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, Cost {:.1f} sec'.format(
                    epoch, phase, epoch_loss, phase, epoch_acc, time.time() - epoch_start
                ))
                self.writer.add_scalar('Loss/{}'.format(phase), epoch_loss, global_step=epoch)
                # 记录验证Accuracy（确保在验证阶段结束时进行）
                self.writer.add_scalar('Accuracy/{}'.format(phase), epoch_acc, global_step=epoch)

                if phase in ['source_val', 'target_val']:
                    all_labels = np.array(all_labels)
                    all_predictions = np.array(all_predictions)
                    all_predictions_labels = np.argmax(all_predictions, axis=1)
                    calculate_and_log_metrics(all_labels, all_predictions_labels)

                # save the model
                if phase == 'target_val':
                    # save the checkpoint for other learning
                    model_state_dic = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
                    # save the best model according to the val accuracy
                    if epoch_acc > best_acc or epoch > args.max_epoch-2:
                        best_acc = epoch_acc
                        logging.info("save best model epoch {}, acc {:.4f}".format(epoch, epoch_acc))
                        torch.save(model_state_dic,
                                   os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)))


            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

def calculate_and_log_metrics(all_labels, all_predictions):
    precision, recall, fscore, support = precision_recall_fscore_support(all_labels, all_predictions, average=None)

    # Logging metrics for each class
    num_classes = len(set(all_labels))  # Determining the number of classes dynamically
    for i in range(num_classes):
        logging.info(
            f'Class {i}: Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1-Score: {fscore[i]:.4f}')

        # Calculating and logging macro-average metrics
    macro_precision, macro_recall, macro_fscore, _ = precision_recall_fscore_support(all_labels, all_predictions,
                                                                                     average='macro')
    logging.info(f'Macro-average Precision: {macro_precision:.4f}')
    logging.info(f'Macro-average Recall: {macro_recall:.4f}')
    logging.info(f'Macro-average F1-Score: {macro_fscore:.4f}')

    cm = confusion_matrix(all_labels, all_predictions)
    logging.info('\nConfusion Matrix (formatted):')
    for row in cm:
        logging.info(' '.join(f'{num:5d}' for num in row))