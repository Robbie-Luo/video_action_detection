import os
import argparse
import numpy as np
from tqdm import tqdm
import glob
import pickle
import cv2
import torch
from torchvision import transforms
import torch.utils.data as data_utl
import videotransforms
from dataset import VidOR
from frames import load_vidor_dataset

class VidorDataset(data_utl.Dataset):

    def __init__(self, dataset_path, split, mode = 'rgb', low_memory=True):
        self.anno_rpath = os.path.join(dataset_path,'annotation')
        self.video_rpath = os.path.join(dataset_path,'video')
        self.frame_rpath = os.path.join(dataset_path,'frame')
        self.split = split
        self.mode = mode
        self.low_memory = low_memory
        self.data = self.make_vidor_data()
        self.max_length = 2
        if split == 'training':
            self.transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip()])
        else:
            self.transforms = transforms.Compose([videotransforms.CenterCrop(224)])
        
    def make_vidor_data(self):
        pkl_path = f'dataset/vidor_{self.split}_data.pkl'
        if not os.path.exists(pkl_path):
            vidor_dataset = load_vidor_dataset()
            vidor_data = []
            actions = vidor_dataset._get_action_predicates()
            vids = vidor_dataset.get_index(self.split)
            for ind in tqdm(vids):
                for each_ins in vidor_dataset.get_action_insts(ind):
                    video_path = vidor_dataset.get_video_path(ind,self.split)
                    start_f, end_f = each_ins['duration']
                    label = np.full((1, end_f - start_f), actions.index(each_ins['category']))
                    vidor_data.append((video_path, label, start_f, end_f))
            with open(pkl_path,'wb') as file:
                pickle.dump(vidor_data,file)
        else:
            with open(pkl_path,'rb') as file:
                vidor_data = pickle.load(file)
        return vidor_data

    
    def load_frames(self, video_path, start, end):
        frame_path = video_path.replace('video','frame').replace('.mp4','')
        frames = sorted(glob.glob(frame_path+'/*.jpg'))
        assert len(frames)>0
        if end - start > self.max_length:
            end = start +  self.max_length
        return np.asarray([self.frame_to_array(frame_path) for frame_path in frames[start:end]],dtype=np.float32)
            

    def frame_to_array(self, frame_path):
        img = cv2.imread(frame_path)[:, :, [2, 1, 0]]
        w, h, c = img.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = (img / 255.) * 2 - 1
        return img

            
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        video_path, label, start_f, end_f = self.data[index]
        frames = self.load_frames(video_path,start_f, end_f)
        frames = self.transforms(frames)
        frames_tensor  = torch.from_numpy(frames.transpose([3, 0, 1, 2]))
        if label.shape[1] > self.max_length:
            label = label[:self.max_length]
        label_one_hot = np.eye(42)[np.squeeze(label)]
        return frames_tensor, torch.from_numpy(label_one_hot)

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
    dataset = VidorDataset(dataset_path, args.split)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=36,
                                                pin_memory=False)

    


