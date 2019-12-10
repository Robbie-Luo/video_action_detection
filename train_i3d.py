import argparse
import os
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

def load_data(dataset_path, batch_size = 1, num_workers=40):
    batch_size = 1
    dataset_train = VidorDataset(dataset_path, 'training')
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                                pin_memory=True)
    dataset_val = VidorDataset(dataset_path, 'validation')
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                                pin_memory=True)
    dataloaders = {'train': dataloader_train, 'val': dataloader_val}
    datasets = {'train': dataset_train, 'val': dataset_val}
    return datasets, dataloaders

def run(dataloaders, save_model='output/', num_steps_per_update = 4, max_steps=64e3, num_classes=42):
    i3d = InceptionI3d(400, in_channels=3)
    i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
    # i3d.replace_logits(157)
    i3d.replace_logits(num_classes)
    i3d.cuda()
    i3d = nn.DataParallel(i3d)
    optimizer = optim.SGD(i3d.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-7)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])
    steps = 0
    # train it
    while steps < max_steps:  # for epoch in range(num_epochs):
        print('Step {}/{}'.format(steps, max_steps))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)  # Set model to evaluate mode

            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()

            for data in tqdm(dataloaders[phase]):
                num_iter += 1
                # get the inputs
                inputs, labels = data
                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                t = inputs.size(2)
                labels = Variable(labels.cuda())
                # print(inputs.size())
                # print(labels.size())

                per_frame_logits = i3d(inputs)
                # upsample to input size
                per_frame_logits = F.interpolate(per_frame_logits, t, mode='linear',align_corners=True)
                a = per_frame_logits.cpu().detach().numpy()
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

                loss = (0.5*loc_loss + 0.5*cls_loss)/num_steps_per_update
                tot_loss += loss.data.item()
                loss.backward()

                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_sched.step()
                    if steps % 10 == 0:
                        print('Step {}/{}'.format(steps, max_steps))
                        print('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss/(10*num_steps_per_update), tot_cls_loss/(10*num_steps_per_update), tot_loss/10))
                        # save model
                        torch.save(i3d.module.state_dict(), save_model+str(steps).zfill(6)+'.pt')
                        tot_loss = tot_loc_loss = tot_cls_loss = 0.
            if phase == 'val':
                print('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss/num_iter, tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter) )
    
if __name__ == '__main__':
    dataset_path = '/home/wluo/vidor-dataset'
    parser = argparse.ArgumentParser()
    # parser.add_argument('-num_epochs', type=int, default=100)
    parser.add_argument('-save_model', type=str, default="output/")
    args = parser.parse_args()
    datasets, dataloaders = load_data(dataset_path)
    run(dataloaders, args.save_model)
