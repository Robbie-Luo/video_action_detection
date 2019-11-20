import argparse
from dataset import VidOR
import torch
import torch.utils.data as data_utl
from utils import *
if __name__ == '__main__':
    dataset_path = '/home/wluo/vidor-dataset'
    split = 'validation'
    extract_frames(dataset_path, split)
