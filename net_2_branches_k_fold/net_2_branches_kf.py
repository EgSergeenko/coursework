#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import warnings
import pickle
import time
from threading import Thread

import wandb
import GPUtil
import numpy as np
from PIL import Image
from skimage import io
from prettytable import PrettyTable

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, average_precision_score

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet34
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


# In[ ]:


warnings.filterwarnings('ignore')


# In[ ]:


wandb.init(project='net_2_branches_kf')


# In[ ]:


DEVICE_ID = 0
if torch.cuda.is_available():
    dev = f'cuda:{DEVICE_ID}'
else:
    dev = 'cpu'
device = torch.device(dev)
print(f'Current device: {dev}')


# In[ ]:


IMAGES_DIR = '/mnt/tank/scratch/esergeenko/net_2_branches/'
LABELS_DISEASES_FILE = 'net_2_branches_diseases.pickle'
LABELS_MORPHOLOGY_FILE = 'net_2_branches_morphology.pickle'


# In[ ]:


wandb.config.lr = 0.0001
wandb.config.batch_size = 32
wandb.config.epochs = 100
wandb.config.fold_index = 0


# In[ ]:


CHANNELS = 3
HEIGHT = 224
WIDTH = 224


# In[ ]:


with open(LABELS_DISEASES_FILE, 'rb') as f:
    labels_diseases = pickle.load(f)
with open(LABELS_MORPHOLOGY_FILE, 'rb') as f:
    labels_morphology = np.array(pickle.load(f))


# In[ ]:


class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.stopped = False
        self.delay = delay
        self.start()

    def run(self):
        while not self.stopped:
            for gpu in GPUtil.getGPUs():
                if gpu.id == 0:
                    print('|'.join([f'{"ID": ^5}', f'{"GPU util.": ^10}', f'{"Memory util.": ^14}', f'{"Memory used": ^14}',
                                    f'{"Memory total": ^14}', f'{"T": ^6}']))
                    print('|'.join([f'{gpu.id: ^5}', f'{f"{int(gpu.load * 100)}%": ^10}', f'{f"{int(gpu.memoryUtil * 100)}%": ^14}',
                                    f'{f"{int(gpu.memoryUsed)}MB": ^14}', f'{f"{int(gpu.memoryTotal)}MB": ^14}',
                                    f'{f"{int(gpu.memoryUsed)}С°": ^6}']))
                                print(gpu_stats)
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True


# In[ ]:


def pretty_print_time(seconds):
    seconds = int(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    print(f'Time spent: {hours}:{minutes}:{seconds}')


# In[ ]:


def timeit(func):
    def timed(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        pretty_print_time(end - start)
        return result
    return timed


# In[ ]:


class CustomCrop:
    
    def __call__(self, sample):
        shape = sample.shape
        min_dimension = min(shape[1], shape[2])
        center_crop = transforms.CenterCrop(min_dimension)
        sample = center_crop(sample)
        return sample


# In[ ]:


primary_transforms = transforms.Compose(
    [transforms.PILToTensor(), CustomCrop(), transforms.Resize((WIDTH, HEIGHT))])


# In[ ]:


@timeit
def get_images(transforms, n, log_step):
    images = torch.empty((n, CHANNELS, WIDTH, HEIGHT), dtype=torch.uint8)
    for i in range(n):
        if (i + 1) % log_step == 0:
            print(f'Images loaded: {i + 1}/{n}')
        image = Image.open(IMAGES_DIR + f'{i}.jpg')
        images[i] = transforms(image).detach().clone()
    print('Images were loaded from the disk.')
    return images


# In[ ]:


images = get_images(primary_transforms, len(labels_diseases), 5000)


# In[ ]:


class Net(nn.Module):
    def __init__(self, resnet, out_features_diseases, out_features_morph):
        super(Net, self).__init__()
        self.base_model = nn.Sequential(*list(resnet.children())[:-1])
        self.branch_1 = nn.Linear(512, out_features_diseases)
        self.branch_2 = nn.Linear(512, out_features_morph)
        
    def forward(self, x):
        x = self.base_model(x)
        x = torch.flatten(x, 1)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        return x1, x2


# In[ ]:


class MorphDisDataset(Dataset):

    def __init__(self, images, labels_diseases, labels_morphology, transform):
        self.images = images
        self.transform = transform
        self.labels_diseases = labels_diseases
        self.labels_morphology = labels_morphology

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = self.images[idx].float() / 255
        labels_morphology = self.labels_morphology[idx]
        labels_diseases = self.labels_diseases[idx]

        if self.transform:
            sample = {'image': self.transform(image), 'labels_diseases': labels_diseases,
                      'labels_morphology': labels_morphology}
        else:
            sample = {'image': image, 'labels_diseases': labels_diseases, 'labels_morphology': labels_morphology}
        return sample


# In[ ]:


secondary_transforms = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
dataset = MorphDisDataset(images, labels_diseases, labels_morphology, secondary_transforms)


# In[ ]:


kf = StratifiedKFold(n_splits=5, random_state=42)
for i, (train_indexes, test_indexes) in enumerate(kf.split(np.arange(len(labels_diseases)), labels_diseases)):
    if i == wandb.config.fold_index:
        train_sampler = SubsetRandomSampler(train_indexes)
        test_sampler = SubsetRandomSampler(test_indexes)


# In[ ]:


train_loader = DataLoader(dataset, batch_size=wandb.config.batch_size, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=wandb.config.batch_size, sampler=test_sampler)


# In[ ]:


pos_weight = torch.tensor(np.sum(labels_morphology, axis=0))
neg_weight = torch.tensor(np.sum(np.invert(labels_morphology.astype(np.bool)).astype(np.int), axis=0))
pos_weight = neg_weight / pos_weight
pos_weight = pos_weight.to(device)


# In[ ]:


net = Net(resnet34(pretrained=True), 20, 8)
criterion_morphology = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
criterion_diseases = nn.CrossEntropyLoss()
net = net.to(device)
optimizer = Adam(net.parameters(), wandb.config.lr)


# In[ ]:


mapping_morphology = [
    'пятно',
    'бугорок',
    'узел',
    'папула',
    'волдырь',
    'пузырек',
    'пузырь',
    'гнойничок'
]


# In[ ]:


def get_labels(predictions, treshold):
    return (predictions > treshold).astype(int)


# In[ ]:


def get_metric_value(metric_name, y_true, y_pred):
    if 'f1' in metric_name:
        return f1_score(y_true, y_pred, average='macro')
    elif 'precision' in metric_name:
        return precision_score(y_true, y_pred, average='macro')
    elif 'recall' in metric_name:
        return recall_score(y_true, y_pred, average='macro')
    elif 'accuracy' in metric_name:
        return accuracy_score(y_true, y_pred)
    elif 'mAP' in metric_name:
        return average_precision_score(y_true, y_pred)


# In[ ]:


def log_epoch_diseases(epoch, y_true_train, y_pred_train, y_true_test, y_pred_test, train_loss, test_loss,
                       current_metrics):
    step = {'epoch': epoch, f'dis loss/train': train_loss, f'dis loss/test': test_loss}

    for k, v in current_metrics.items():
        if 'train' in k:
            value = get_metric_value(k, y_true_train, y_pred_train)
        else:
            value = get_metric_value(k, y_true_test, y_pred_test)
        step[k] = value
        current_metrics[k] = max(value, v)
    return current_metrics, step


def log_epoch_morphology(step, y_true_train, y_pred_train, y_true_test, y_pred_test, train_loss, test_loss, mapping,
                         current_metrics):
    step['mor loss/train'] = train_loss
    step['mor loss/test'] = test_loss

    for k, v in current_metrics.items():
        if 'class' in k:
            i = mapping.index(k.split('/')[1].split(' ')[1])
            if 'train' in k:
                value = get_metric_value(k, y_true_train[:, i], y_pred_train[:, i])
            else:
                value = get_metric_value(k, y_true_test[:, i], y_pred_test[:, i])
        elif 'mAP' in k:
            if 'train' in k:
                value = get_metric_value(k, y_true_train.reshape(-1), y_pred_train.reshape(-1))
            else:
                value = get_metric_value(k, y_true_test.reshape(-1), y_pred_test.reshape(-1))
        else:
            treshold = float(k.split('/')[1].split(' ')[1])
            if 'train' in k:
                value = get_metric_value(k, y_true_train, get_labels(y_pred_train, treshold))
            else:
                value = get_metric_value(k, y_true_test, get_labels(y_pred_test, treshold))
        step[k] = value
        current_metrics[k] = max(v, value)

    wandb.log(step)
    return current_metrics


# In[ ]:


@timeit
def evaluate_net(net, data_loader, log_step, data_loader_name='train'):
    net.eval()
    with torch.no_grad():
        y_true_b1 = np.zeros((1))
        y_pred_b1 = np.empty((1))
        y_true_b2 = np.empty((1, 8))
        y_pred_b2 = np.empty((1, 8))
        loss_b1 = 0.0
        loss_b2 = 0.0
        print(f'Evaluating {data_loader_name}:')
        for i, data in enumerate(data_loader, 0):
            inputs, labels_diseases, labels_morphology = data['image'], data['labels_diseases'], data[
                'labels_morphology']
            o1, o2 = net(inputs.to(device))

            mask = torch.tensor(1 - (labels_morphology.sum(axis=1) > 0) * 1).int()

            labels_morphology = labels_morphology[mask != 1].to(device)
            o2 = o2[mask != 1]

            loss1 = criterion_diseases(o1, labels_diseases.to(device).long())
            loss2 = criterion_morphology(o2, labels_morphology)

            loss_b1 += loss1.item()
            loss_b2 += loss2.item()

            predicted_b1 = torch.argmax(F.softmax(o1), axis=1).cpu().detach().numpy()
            predicted_b2 = torch.sigmoid(o2).cpu().detach().numpy()

            y_true_b1 = np.concatenate((y_true_b1, labels_diseases.cpu().numpy()))
            y_pred_b1 = np.concatenate((y_pred_b1, predicted_b1))

            y_true_b2 = np.concatenate((y_true_b2, labels_morphology.cpu().numpy()))
            y_pred_b2 = np.concatenate((y_pred_b2, predicted_b2))

            if (i + 1) % log_step == 0:
                print(f'Batches processed: {i + 1}/{len(data_loader)}')

        loss_b1 = loss_b1 / len(data_loader)
        loss_b2 = loss_b2 / len(data_loader)

        y_true_b1 = y_true_b1[1:].astype(np.int)
        y_pred_b1 = y_pred_b1[1:]
        y_true_b2 = y_true_b2[1:].astype(np.int)
        y_pred_b2 = y_pred_b2[1:]

        return (y_true_b1, y_pred_b1, loss_b1), (y_true_b2, y_pred_b2, loss_b2)


# In[ ]:


@timeit
def train_net(net, optimizer, data_loader, log_step):
    net.train()
    print('Training:')
    for i, data in enumerate(data_loader, 0):
        inputs, labels_diseases, labels_morphology = data['image'], data['labels_diseases'], data['labels_morphology']

        optimizer.zero_grad()

        o1, o2 = net(inputs.to(device))

        loss1 = criterion_diseases(o1, labels_diseases.to(device).long())

        mask = torch.tensor(1 - (labels_morphology.sum(axis=1) > 0) * 1).int()
        labels_morphology = labels_morphology[mask != 1].to(device)
        o2 = o2[mask != 1]
        loss2 = criterion_morphology(o2, labels_morphology)

        if torch.isnan(loss2).any():
            loss = loss1
        else:
            loss = loss1 + loss2

        if (i + 1) % log_step == 0:
            print(f'Batches processed: {i + 1}/{len(data_loader)}')

        loss.backward()
        optimizer.step()


# In[ ]:


def initialize_metrics(dis_prefix, mor_prefix, mapping):
    d_metrics_list = ['f1', 'recall', 'accuracy', 'precision']
    metrics_diseases = {}
    for m in d_metrics_list:
        metrics_diseases[f'{dis_prefix} {m}/train'] = 0
        metrics_diseases[f'{dis_prefix} {m}/test'] = 0
    m_metrics_list = ['f1', 'recall', 'precision']
    metrics_morphology = {}
    for m in m_metrics_list:
        for treshold in np.arange(0.1, 1, 0.1):
            metrics_morphology[f'{mor_prefix} {m}/train {round(treshold, 1)}'] = 0
            metrics_morphology[f'{mor_prefix} {m}/test {round(treshold, 1)}'] = 0
    for i in range(8):
        metrics_morphology[f'{mor_prefix} mAP class/train {mapping[i]}'] = 0
        metrics_morphology[f'{mor_prefix} mAP class/test {mapping[i]}'] = 0
    metrics_morphology[f'mor mAP/train'] = 0
    metrics_morphology[f'mor mAP/test'] = 0
    return metrics_diseases, metrics_morphology


# In[ ]:


monitor = Monitor(600)
metrics_diseases, metrics_morphology = initialize_metrics('dis', 'mor', mapping_morphology)
for epoch in range(wandb.config.epochs):
    train_net(net, optimizer, train_loader, 250)
    b1, b2 = evaluate_net(net, train_loader, 250, 'train')
    y_true_train_b1, y_pred_train_b1, loss_train_b1 = b1
    y_true_train_b2, y_pred_train_b2, loss_train_b2 = b2
    b1, b2 = evaluate_net(net, test_loader, 100, 'test')
    y_true_test_b1, y_pred_test_b1, loss_test_b1 = b1
    y_true_test_b2, y_pred_test_b2, loss_test_b2 = b2 
    metrics_diseases, step = log_epoch_diseases(epoch, y_true_train_b1, y_pred_train_b1, y_true_test_b1, y_pred_test_b1,
                                                loss_train_b1, loss_test_b1, metrics_diseases)
    metrics_morphology = log_epoch_morphology(step, y_true_train_b2, y_pred_train_b2, y_true_test_b2, y_pred_test_b2,
                                              loss_train_b2, loss_test_b2, mapping_morphology, metrics_morphology)
print('Finished')
monitor.stop()


# In[ ]:


for k, v in metrics_diseases.items():
    wandb.run.summary[k] = v
for k, v in metrics_morphology.items():
    wandb.run.summary[k] = v

