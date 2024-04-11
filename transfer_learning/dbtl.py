#!/usr/bin/python
# -*- coding:utf-8 -*-
import logging
import warnings
import os
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from itertools import chain
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from utils.BatchIdxDataset import BatchIdxDataset

import models, datasets
from torch.utils.tensorboard import SummaryWriter

def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.eval()

class dbtl:
    def __init__(self, args, save_dir):
        self.metrics_log = []
        self.args = args
        self.save_dir = save_dir
        self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, 'tensorboard_logs'))
        self.num_classes = None
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

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
        self.datasets['source_train'], self.datasets['source_val'], self.datasets['target_train'], self.datasets['target_val'] = dataset.data_split(
            args.gear_health_check, transfer_learning=True)

        self.n_source = len(self.datasets['source_train'])
        self.n_target = len(self.datasets['target_train'])
        self.n_target = len(self.datasets['target_val'])

        print(self.n_source)
        print(self.n_target)

        # concat_dataset = ConcatDataset([self.datasets['source_train'], self.datasets['target_train']])

        self.datasets['train'] = BatchIdxDataset([self.datasets['source_train'], self.datasets['target_train']])
        self.datasets['val'] = self.datasets['target_val']
        print(len(self.datasets['val']))
        # print(self.datasets['val'].lengths)
        self.dataloaders = {
            'train': torch.utils.data.DataLoader(self.datasets['train'], batch_size=self.args.batch_size,
                                                        shuffle=True, num_workers=self.args.num_workers,
                                                        pin_memory=True),
            'val': torch.utils.data.DataLoader(self.datasets['val'], batch_size=self.args.batch_size,
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


        if args.model_name == 'CNN':
            layers_to_freeze = [self.model.layer1, self.model.layer2, self.model.layer3]
            for layer in layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
        elif args.model_name == 'MLP':
            for layer in [self.model.fc1, self.model.fc2, self.model.fc3, self.model.fc4, self.model.fc5]:
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

        self.model.to(self.device)

        # if args.adabn:
        #     self.model_eval.to(self.device)


    def train(self):

        weights = np.concatenate((np.ones(self.n_source) / self.n_source, np.ones(self.n_target) / self.n_target))
        beta = 1 / (1 + np.sqrt(2 * np.log(self.n_source) / self.args.max_epoch))
        best_acc = 0.0
        # self.criterion = nn.CrossEntropyLoss(weight=weights)
        for epoch in range(self.args.max_epoch):
            print(
                f"Epoch {epoch}: Max weight={np.max(weights)}, Min weight={np.min(weights)}, Mean weight={np.mean(weights)}")
            all_labels = []
            all_predictions = []
            epoch_labels = []
            epoch_preds = []
            epoch_indices = []
            logging.info(f'Epoch {epoch}/{self.args.max_epoch - 1}')
            self.model.train()
            total_loss = 0.0
            correct_predictions = 0
            normalize_weights(weights)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=1e-4)
            # train
            for inputs, labels, indices in self.dataloaders['train']:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                indices = indices.cpu().numpy()

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                print(np.max(weights))
                current_weights = torch.tensor(weights[indices], dtype=torch.float32, device=self.device)

                losses = F.cross_entropy(outputs, labels, reduction='none')
                weighted_losses = losses * current_weights
                loss = weighted_losses.mean()

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_predictions += torch.sum(preds == labels.data).item()

                # epoch_labels.append(labels.cpu().numpy())
                # epoch_preds.append(preds.cpu().numpy())
                # epoch_indices.append(indices)
                epoch_labels.extend(labels.cpu().tolist())
                epoch_preds.extend(preds.cpu().tolist())
                epoch_indices.extend(indices.tolist())

            epoch_labels = np.array(epoch_labels)
            epoch_preds = np.array(epoch_preds)
            epoch_indices = np.array(epoch_indices)


            epoch_loss = total_loss / len(self.dataloaders['train'].dataset)
            epoch_acc = correct_predictions / len(self.dataloaders['train'].dataset)
            logging.info(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            beta_e = calculate_beta_e(weights, epoch_indices, epoch_preds, epoch_labels, self.n_source)
            weights = update_weights(weights, epoch_preds, epoch_indices, epoch_labels, beta, beta_e, self.n_source)

            # val
            self.model.eval()
            val_loss = 0.0
            val_corrects = 0
            with torch.no_grad():
                for inputs, labels in self.dataloaders['val']:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = F.cross_entropy(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    val_corrects += torch.sum(preds == labels.data).item()

                    all_labels.append(labels.cpu().numpy())
                    all_predictions.append(preds.cpu().numpy())

            val_epoch_loss = val_loss / len(self.dataloaders['val'].dataset)
            val_epoch_acc = val_corrects / len(self.dataloaders['val'].dataset)
            logging.info(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')


            if val_epoch_acc > best_acc:
                best_acc = val_epoch_acc
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_model.pth'))
                logging.info('Saved best model')
                all_labels = np.concatenate(all_labels)
                all_predictions = np.concatenate(all_predictions)
                calculate_and_log_metrics(all_labels, all_predictions, self.num_classes)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        logging.info(f'Best Accuracy: {best_acc:.4f}')





# def update_weights(weights, predictions, indices, true_labels, beta, beta_e, n_source):
#     for i, idx in enumerate(indices):
#         pred = predictions[i]
#         true = true_labels[i]
#         if idx < n_source:  # 源域样本
#             if pred != true:
#                 weights[idx] *= beta
#             else:
#                 weights[idx] *= (1 - beta)
#         else:  # 目标域样本
#             if pred != true:
#                 weights[idx] *= beta_e
#             else:
#                 weights[idx] /= beta_e
#
#     weights_exp = np.exp(weights - np.max(weights))  # Softmax 归一化
#     weights = weights_exp / np.sum(weights_exp)
#     return weights
def update_weights(weights, predictions, indices, true_labels, beta, beta_e, n_source):
    for i, idx in enumerate(indices):
        pred = predictions[i]
        true = true_labels[i]
        error_sign = -1 if pred == true else 1
        if idx < n_source:
            weights[idx] *= np.exp(beta * error_sign)
        else:
            weights[idx] *= np.exp(beta_e * error_sign)

    # 防止权重过小或过大，重新归一化
    weights /= np.sum(weights)
    return weights


def calculate_beta_e(weights, indices, predictions, labels, n_source):
    indices = np.array(indices)
    predictions = np.array(predictions)
    labels = np.array(labels)

    target_indices = indices[indices >= n_source] - n_source
    target_errors = predictions[target_indices] != labels[target_indices]
    error_e = np.sum(weights[target_indices][target_errors]) / np.sum(weights[target_indices])
    beta_e = error_e / (1.0 - error_e) if error_e < 1.0 else 1.0
    return beta_e


def normalize_weights(weights):
    total_weight = np.sum(weights)
    weights = weights / total_weight
    weights = np.clip(weights, 1e-4, 1)
    return weights



def calculate_and_log_metrics(all_labels, all_predictions, num_classes):
    # all_labels = all_labels.cpu().numpy() if all_labels.is_cuda else all_labels.numpy()
    # all_predictions = all_predictions.cpu().numpy() if all_predictions.is_cuda else all_predictions.numpy()

    precision, recall, fscore, support = precision_recall_fscore_support(all_labels, all_predictions, average=None)

    num_classes = len(set(all_labels))
    for i in range(num_classes):
        logging.info(
            f'Class {i}: Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1-Score: {fscore[i]:.4f}')

    macro_precision, macro_recall, macro_fscore, _ = precision_recall_fscore_support(all_labels, all_predictions,
                                                                                     average='macro')
    logging.info(f'Macro-average Precision: {macro_precision:.4f}')
    logging.info(f'Macro-average Recall: {macro_recall:.4f}')
    logging.info(f'Macro-average F1-Score: {macro_fscore:.4f}')
    cm = confusion_matrix(all_labels, all_predictions)

    logging.info('\nConfusion Matrix (formatted):')
    for row in cm:
        logging.info(' '.join(f'{num:5d}' for num in row))

