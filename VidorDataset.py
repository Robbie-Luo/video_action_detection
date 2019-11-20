import os
import argparse
import numpy as np
from tqdm import tqdm
import glob
import cv2
import torch
import torch.utils.data as data_utl

from dataset import VidOR
from utils import load_vidor_dataset

class VidorDataset(data_utl.Dataset):

    def __init__(self, dataset_path, split, mode = 'rgb', low_memory=True):
        self.anno_rpath = os.path.join(dataset_path,'annotation')
        self.video_rpath = os.path.join(dataset_path,'video')
        self.frame_rpath = os.path.join(dataset_path,'frame')
        self.split = split
        self.mode = mode
        self.low_memory = low_memory
        self.data = self.make_vidor_data()
        
    def make_vidor_data(self):
        vidor_dataset = load_vidor_dataset()
        vidor_data = []
        # with open('actions.json', 'r') as action_f:
        #     actions = json.load(action_f)['actions']
        actions = vidor_dataset._get_action_predicates()
        vids = vidor_dataset.get_index(self.split)
        for ind in tqdm(vids):
            for each_ins in vidor_dataset.get_action_insts(ind):
                video_path = vidor_dataset.get_video_path(ind,self.split)
                start_f, end_f = each_ins['duration']
                label = np.full((1, end_f - start_f), actions.index(each_ins['category']))
                vidor_data.append((video_path, label, start_f, end_f))
        return vidor_data
    
    def load_frames(self, video_path, start, end):
        frame_path = video_path.replace('video','frame').replace('.mp4','')
        frames = sorted(glob.glob(frame_path+'/*.jpg'))
        assert len(frames)>0
        return np.array([self.frame_to_array(frame_path) for frame_path in frames[start:end]])
            

    def frame_to_array(self, frame_path):
        img = cv2.imread(frame_path)[:, :, [2, 1, 0]]
        w, h, c = img.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = (img / 255.) * 2 - 1
        return np.asarray(img, dtype=np.float32)

            
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        video_path, label, start_f, end_f = self.data[index]
        frames = self.load_frames(video_path,start_f, end_f)
        frames_tensor  = torch.from_numpy(frames .transpose([3, 0, 1, 2]))
        return frames_tensor, torch.from_numpy(label)

    def __len__(self):
        return len(self.data)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-split', type=str, default="validation")
    args = parser.parse_args()

    dataset_path ='/home/wluo/vidor-dataset'
    mode = 'rgb'
    task = 'action'
    save_dir = 'output/features/'
    low_memory = True
    batch_size = 1
    dataset = VidorDataset(dataset_path, args.split)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=36,
                                                pin_memory=False)

    


