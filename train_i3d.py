import argparse
import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utl
from torch.autograd import Variable
from torchvision import transforms
from pytorch_i3d import InceptionI3d
from VidorDataset import VidorDataset
from tqdm import tqdm
from util import *

DATASET_LOC = '/home/wluo/vidor-dataset'
TRAIN_LOG_LOC = 'output/train.log'

def load_data(dataset_path, num_workers=100):
    dataset_train = VidorDataset(dataset_path, 'training')
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=num_workers,
                                                pin_memory=True)
    dataset_val = VidorDataset(dataset_path, 'validation')
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=num_workers,
                                                pin_memory=True)
    dataloaders = {'train': dataloader_train, 'val': dataloader_val}
    datasets = {'train': dataset_train, 'val': dataset_val}
    return datasets, dataloaders

def print_log(line):
    logging.info(line)
    print(line)

def run(dataloaders, save_model='output/', num_epochs = 10, num_per_update = 5, num_classes=42):
    i3d = InceptionI3d(400, in_channels=3)
    i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
    # i3d.load_state_dict(torch.load('output/120000.pt'))
    i3d.replace_logits(num_classes)
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    optimizer = optim.SGD(i3d.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-7)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])
    iteration = 0
    max_iterations = num_epochs * len(dataloaders['train'])
    for epoch in range(num_epochs):
        # Training phase
        print_statement('MODEL TRAINING')
        tot_loss = 0.0
        tot_loc_loss = 0.0
        tot_cls_loss = 0.0
        for data in tqdm(dataloaders['train']):
            iteration+=1
            # get the inputs
            inputs, labels = data
            # wrap them in Variable
            inputs = Variable(inputs.cuda())
            t = inputs.size(2)
            labels = Variable(labels.cuda())
            # set the model to train
            i3d.train(True)
            optimizer.zero_grad()
            per_frame_logits = i3d(inputs)
            # upsample to input size
            per_frame_logits = F.interpolate(per_frame_logits, t, mode='linear',align_corners=True)
            # unified dimension
            if per_frame_logits.size() != labels.size():
                labels.resize_(per_frame_logits.size())
            labels = labels.type(torch.cuda.FloatTensor)
            # compute localization loss
            loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
            tot_loc_loss += loc_loss.data.item()
            # compute classification loss (with max-pooling along time B x C x T)
            cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
            tot_cls_loss += cls_loss.data.item()
            # compute total loss
            loss = (0.5*loc_loss + 0.5*cls_loss)
            tot_loss += loss.data.item()
            loss.backward()
            if iteration % num_per_update == 0:
                optimizer.step()
                lr_sched.step()
                if iteration % 100 == 0:
                    print_log('Train step:{:d}/{:d},Loc Loss: {:.4f},Cls Loss: {:.4f},Tot Loss: {:.4f}'
                          .format(iteration,max_iterations,tot_loc_loss/num_per_update,tot_cls_loss/num_per_update,tot_loss/num_per_update))
                if iteration % 1000 == 0:
                    torch.save(i3d.module.state_dict(), save_model+str(iteration).zfill(6)+'.pt')
                tot_loss = tot_loc_loss = tot_cls_loss = 0.
            if iteration % 10000 == 0:
            
                # Validation phase
                print_statement('MODEL VALIDATING')
                # set the model to evaluate
                i3d.train(False)
                optimizer.zero_grad()
                tot_loss = tot_loc_loss = tot_cls_loss = 0.
                num_iter = 0 
                for data in tqdm(dataloaders['val']):
                    num_iter+=1
                    # get the inputs
                    inputs, labels = data
                    # wrap them in Variable
                    inputs = Variable(inputs.cuda())
                    t = inputs.size(2)
                    labels = Variable(labels.cuda())

                    per_frame_logits = i3d(inputs)
                    # upsample to input size
                    per_frame_logits = F.interpolate(per_frame_logits, t, mode='linear',align_corners=True)
                    # unified dimension
                    if per_frame_logits.size() != labels.size():
                        labels.resize_(per_frame_logits.size())
                    labels = labels.type(torch.cuda.FloatTensor)
                    # compute localization loss
                    loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                    tot_loc_loss += loc_loss.data.item()
                    # compute classification loss (with max-pooling along time B x C x T)
                    cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
                    tot_cls_loss += cls_loss.data.item()
                    # compute total loss
                    loss = (0.5*loc_loss + 0.5*cls_loss)
                    tot_loss += loss.data.item()
                print_log('Validation: Loc Loss: {:.4f},Cls Loss: {:.4f},Tot Loss: {:.4f}'
                        .format(tot_loc_loss/num_iter,tot_cls_loss/num_iter,tot_loss/num_iter))

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-num_epochs', type=int, default=100)
    parser.add_argument('-save_model', type=str, default="output/")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG, filename="output/train.log", filemode="w+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    datasets, dataloaders = load_data(DATASET_LOC)
    run(dataloaders, args.save_model)
