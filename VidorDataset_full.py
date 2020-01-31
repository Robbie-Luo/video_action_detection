import os
import argparse
import numpy as np
from tqdm import tqdm
import glob
import pickle
import random
import cv2
import torch
from torchvision import transforms
import torch.utils.data as data_utl
import videotransforms
from dataset import VidOR
from frames import load_vidor_dataset

class VidorDataset(data_utl.Dataset):

    def __init__(self, dataset_path, split, transforms, mode = 'rgb', low_memory=True, num_classes = 42):
        self.anno_rpath = os.path.join(dataset_path,'annotation')
        self.video_rpath = os.path.join(dataset_path,'video')
        self.frame_rpath = os.path.join(dataset_path,'frame')
        self.split = split
        self.mode = mode
        self.low_memory = low_memory
        self.num_classes = num_classes
        self.data = self.make_vidor_data()
        self.transforms = transforms
        
    def make_vidor_data(self):
        pkl_path = f'dataset/vidor_{self.split}_data.pkl'
        if not os.path.exists(pkl_path):
            vidor_dataset = load_vidor_dataset()
            vidor_data = []
            actions = vidor_dataset._get_action_predicates()
            vids = vidor_dataset.get_index(self.split)
            for ind in tqdm(vids):
                video_path = vidor_dataset.get_video_path(ind,self.split)
                frame_dir = video_path.replace('video','frame').replace('.mp4','')
                num_frames = len(os.listdir(frame_dir))
                if num_frames < 66:
                    continue
                label = np.zeros((self.num_classes,num_frames), np.float32)
                for each_ins in vidor_dataset.get_action_insts(ind):
                    start_f, end_f = each_ins['duration']
                    action = actions.index(each_ins['category'])
                    for fr in range(0,num_frames,1):
                        if fr >= start_f and fr <= end_f:
                            label[action, fr] = 1 # binary classification
                vidor_data.append((frame_dir,label,num_frames))
            with open(pkl_path,'wb') as file:
                pickle.dump(vidor_data,file)
        else:
            with open(pkl_path,'rb') as file:
                vidor_data = pickle.load(file)
        return vidor_data

    
    def load_rgb_frames(self, frame_dir, start, num):
        # frames = sorted(glob.glob(frame_path+'/*.jpg'))
        def get_image(i):
            img_path = os.path.join(frame_dir, str(i).zfill(4)+'.jpg')
            img = cv2.imread(os.path.join(frame_dir, str(i).zfill(4)+'.jpg'))[:, :, [2, 1, 0]]
            w,h,c = img.shape
            if w < 226 or h < 226:
                d = 226.-min(w,h)
                sc = 1+d/min(w,h)
                img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
            img = (img/255.)*2 - 1
            return img
        frames = [get_image(i) for i in range(start, start+num)] 
        return np.asarray(frames, dtype=np.float32)
            
            
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        frame_dir, label, nf = self.data[index]

        feature_path =  frame_dir.replace('frame','feature')
        if os.path.exists(feature_path+'/i3d_040'+'.npy'):
            return 0,0, feature_path , nf
        if nf > 1000:
            return 0,0, feature_path, nf

        imgs = self.load_rgb_frames(frame_dir, 1, nf)

        imgs = self.transforms(imgs)

        frames_tensor  = torch.from_numpy(imgs.transpose([3, 0, 1, 2]))

        
        return frames_tensor, torch.from_numpy(label), feature_path, nf

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

    


