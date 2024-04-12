#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging
import warnings
import os
import time
import torch
from torch import nn
from torch import optim
import numpy as np

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


import models, datasets
from torch.utils.tensorboard import SummaryWriter

def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.eval()

class fine_tuning:
    def __init__(self, args, save_dir):
        self.metrics_log = []
        self.args = args
        self.save_dir = save_dir
        self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, 'tensorboard_logs'))
        self.num_classes = None

    def setup(self):
        args = self.args
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

        # Load the dataset
        Dataset = getattr(datasets, args.data_name)
        dataset = Dataset(data_dir=args.data_dir, gear_health_check=args.gear_health_check,
                          normlizetype=args.normlizetype)

        # Load data
        self.datasets = {}
        _, _, self.datasets['target_train'], self.datasets['target_val'] = dataset.data_split(
            args.gear_health_check, transfer_learning=True)
        self.dataloaders = {
            'target_train': torch.utils.data.DataLoader(self.datasets['target_train'], batch_size=self.args.batch_size,
                                                        shuffle=True, num_workers=self.args.num_workers,
                                                        pin_memory=True),
            'target_val': torch.utils.data.DataLoader(self.datasets['target_val'], batch_size=self.args.batch_size,
                                                      shuffle=False, num_workers=self.args.num_workers,
                                                      pin_memory=True)
        }
        self.num_classes = 8
        self.prepare_model(args, dataset)

    def prepare_model(self, args, dataset):

        self.model = getattr(models, args.model_name)(args.pretrained==False)
        if (args.model_name == 'MLP'):
            # self.model.classifier = nn.Linear(10, dataset.num_classes)
            self.model.fc = nn.Linear(self.model.classifier.in_features, dataset.num_classes)
        else:
            self.model.fc = torch.nn.Linear(self.model.fc.in_features, dataset.num_classes)

        # if args.adabn:
        #     self.model_eval.to(self.device)

        # checkpoint_path = os.getcwd() + '\checkpoint' + '\\' + args.checkpoint_name
        checkpoint_path = ("F:\\Computational Engineering\\MT\\code\\mt-transfer-learning\\"
                           "checkpoint\\cnn_4layers\\56-0.1577-best_model.pth")
        print(checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint)
        logging.info(f"Loaded pretrained weights from {checkpoint_path}")

        if args.model_name == 'CNN':
            layers_to_freeze = [self.model.layer1, self.model.layer2, self.model.layer3]
            for layer in layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
        elif args.model_name == 'MLP':
            for layer in [self.model.fc1, self.model.fc2, self.model.fc3, self.model.fc4, self.model.fc5]:
                for param in layer.parameters():
                    param.requires_grad = False
        elif args.model_name == 'CNN_3LAYERS':
            layers_to_freeze = [self.model.layer1]
            for layer in layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
        elif args.model_name == 'CNN_4LAYERS':
            layers_to_freeze = [self.model.layer1, self.model.layer2, self.model.layer3]
            for layer in layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False

        self.model.to(self.device)

        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)
            # if args.adabn:
            #     self.model_eval = torch.nn.DataParallel(self.model_eval)

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

        # 将模型移到正确的设备
        self.model.to(self.device)

        # if args.adabn:
        #     self.model_eval.to(self.device)
        self.criterion = nn.CrossEntropyLoss()


    def train(self):
        """
        Training process
        :return:
        """
        args = self.args
        step = 0
        best_acc = 0.0

        for epoch in range(args.max_epoch):
            logging.info('-' * 10)
            logging.info(f'Epoch {epoch}/{args.max_epoch - 1}')
            if self.lr_scheduler is not None:
                # self.lr_scheduler.step(epoch)
                logging.info('current lr: {}'.format(self.lr_scheduler.get_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))

            # Iterate over each phase: training and validation
            for phase in ['target_train', 'target_val']:
                epoch_start = time.time()

                if phase == 'target_train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                all_labels = []
                all_predictions = []

                # Iterate over data.
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # Zero the parameter gradients
                    self.optimizer.zero_grad()

                    # Forward
                    # Track history if only in train
                    with torch.set_grad_enabled(phase == 'target_train'):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)

                        # Backward + optimize only if in training phase
                        if phase == 'target_train':
                            loss.backward()
                            self.optimizer.step()

                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    running_corrects += torch.sum(preds == labels.data)

                    all_labels.append(labels.cpu().numpy())
                    all_predictions.append(outputs.cpu().detach().numpy())

                cost_time = time.time() - epoch_start
                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(self.dataloaders[phase].dataset)

                logging.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Cost Time:{cost_time}')

                # Deep copy the model
                if phase == 'target_val' :


                    all_labels = np.concatenate(all_labels)
                    all_predictions = np.concatenate(all_predictions)
                    all_predictions_labels = np.argmax(all_predictions, axis=1)
                    calculate_and_log_metrics(all_labels, all_predictions_labels, self.num_classes)


                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        # best_model_wts = copy.deepcopy(self.model.state_dict())
                        torch.save(self.model.state_dict(),
                                   os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)))

                self.writer.add_scalar('Loss/{}'.format(phase), epoch_loss, global_step=epoch)

                self.writer.add_scalar('Accuracy/{}'.format(phase), epoch_acc, global_step=epoch)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            logging.info(f'Best val Acc: {best_acc:.4f}')


def calculate_and_log_metrics(all_labels, all_predictions, num_classes):
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
    # logging.info('Confusion Matrix:')
    # logging.info(cm)

    # Optionally, you can print the confusion matrix more prettily
    logging.info('\nConfusion Matrix (formatted):')
    for row in cm:
        logging.info(' '.join(f'{num:5d}' for num in row))


