import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from datasets.SequenceDatasets import dataset
# from datasets.sequence_aug import *
from SequenceDatasets import dataset
from sequence_aug import *
from tqdm import tqdm


signal_size = 1024


# Case One
Case1 = ['helical 1',"helical 2","helical 3","helical 4","helical 5","helical 6"]

label1 = [i for i in range(6)]
#Case Two
Case2 = ['spur 1',"spur 2","spur 3","spur 4","spur 5","spur 6","spur 7","spur 8"]

label2 = [i for i in range(8)]

def map_health_status(case, label):

    healthy_labels_helical = [1, 4, 6]
    healthy_labels_spur = [1, 7, 8]

    # "0" for healthy, "1" for unhealthy
    if "helical" in case and (label + 1) in healthy_labels_helical:
        return 0
    elif "spur" in case and (label + 1) in healthy_labels_spur:
        return 0
    else:
        return 1

def gear_get_files(root, case_list, labels_list):
    data = []
    lab = []
    freq_load_combinations = {
        "30hz": ["High", "Low"],
        "35hz": ["High", "Low"],
        "40hz": ["High", "Low"],
        "45hz": ["High", "Low"],
        "50hz": ["High", "Low"],
    }
    for case_index, case in enumerate(case_list):
        case_path = os.path.join(root, case)
        print(case_path)

        for freq, loads in freq_load_combinations.items():
            for load in loads:
                selected_suffix = random.choice(["1", "2"])
                for suffix in selected_suffix:
                    file_name = f"{case}_{freq}_{load}_{suffix}.txt"
                    file_path = os.path.join(case_path, file_name)
                    if os.path.exists(file_path):
                        health_status = map_health_status(case, labels_list[case_index])
                        data1, lab1 = data_load(file_path, health_status)
                        data.extend(data1)
                        lab.extend(lab1)
                    else:
                        print(f"File does not exist: {file_path}")

    return data, lab


def get_files(root, case_list, labels_list):
    data = []
    lab = []
    freq_load_combinations = {
        "30hz": ["High", "Low"],
        "35hz": ["High", "Low"],
        "40hz": ["High", "Low"],
        "45hz": ["High", "Low"],
        "50hz": ["High", "Low"],
    }
    for case_index, case in enumerate(case_list):
        case_path = os.path.join(root, case)
        print(case_path)

        for freq, loads in freq_load_combinations.items():
            for load in loads:

                selected_suffix = random.choice(["1", "2"])
                for suffix in selected_suffix:
                    file_name = f"{case}_{freq}_{load}_{suffix}.txt"
                    file_path = os.path.join(case_path, file_name)
                    if os.path.exists(file_path):
                        # print(f"Selected file: {file_path}")
                        data1, lab1 = data_load(file_path, labels_list[case_index])
                        data.extend(data1)
                        lab.extend(lab1)
                    else:
                        print(f"File does not exist: {file_path}")

    return data, lab

def data_load(filename,label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    '''
    fl = np.loadtxt(filename,usecols=0)
    fl = fl.reshape(-1,)
    data=[]
    lab=[]
    start,end=0,signal_size
    while end<=fl.shape[0]:
        x = fl[start:end]
        x = np.fft.fft(x)
        x = np.abs(x) / len(x)
        x = x[range(int(x.shape[0] / 2))]
        x = x.reshape(-1, 1)
        data.append(x)
        lab.append(label)
        start +=signal_size
        end +=signal_size

    return data, lab

#--------------------------------------------------------------------------------------------------------------------
class PHMFFT(object):

    inputchannel = 1

    def __init__(self, data_dir, gear_health_check, normlizetype="0-1"):
        self.gear_health_check = gear_health_check
        self.data_dir = data_dir
        # self.source_N = transfer_task[0]
        # self.target_N = transfer_task[1]
        self.normlizetype = normlizetype
        self.data_transforms = {
            'train': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                # RandomAddGaussian(),
                # RandomScale(),
                # RandomStretch(),
                # RandomCrop(),
                Retype(),
                # Scale(1)
            ]),
            'val': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                Retype(),
                # Scale(1)
            ])
        }
        if gear_health_check:
            self.num_classes = 2
        else:
            self.num_classes = 8


    def data_split(self, gear_health_check, transfer_learning=True):
        if transfer_learning:
            # get source train and val
            if (gear_health_check):
                source_data, source_labels = gear_get_files(self.data_dir, Case1, label1)
            else:
                source_data, source_labels = get_files(self.data_dir, Case1, label1)

            source_pd = pd.DataFrame({"data": source_data, "label": source_labels})
            train_pd, val_pd = train_test_split(source_pd, train_size=0.2, test_size=0.8, random_state=40, stratify=source_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            if (gear_health_check):
                target_data, target_labels = gear_get_files(self.data_dir, Case2, label2)
            else:
                target_data, target_labels = get_files(self.data_dir, Case2, label2)

            target_pd = pd.DataFrame({"data": target_data, "label": target_labels})
            train_pd, val_pd = train_test_split(target_pd, train_size=0.2, test_size=0.8, random_state=40, stratify=target_pd["label"])
            target_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            target_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])
            return source_train, source_val, target_train, target_val
        else:

            # print("source")
            if (gear_health_check):
                source_data, source_labels = gear_get_files(self.data_dir, Case1, label1)
            else:
                source_data, source_labels = get_files(self.data_dir, Case1, label1)

            source_pd = pd.DataFrame({"data": source_data, "label": source_labels})

            train_pd, val_pd = train_test_split(source_pd, test_size=0.2, random_state=40, stratify=source_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # Get target dataset
            # print("target")
            if (gear_health_check):
                target_data, target_labels = gear_get_files(self.data_dir, Case2, label2)
            else:
                target_data, target_labels = get_files(self.data_dir, Case2, label2)

            target_pd = pd.DataFrame({"data": target_data, "label": target_labels})
            train_pd, val_pd = train_test_split(target_pd, test_size=0.2, random_state=40, stratify=target_pd["label"])
            target_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            return source_train, source_val, target_val