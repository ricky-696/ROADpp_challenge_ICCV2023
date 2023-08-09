import os
import glob
from tqdm import tqdm
from collections import deque

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision.transforms as transforms

# old cls
"""
class action_dataset(nn.Module):
    def __init__(self, args) -> None:
        datapath = args.datapath

        videos = glob.glob(os.path.join(datapath, "/*/"))

        self.meta_data = {}
        self.meta_tube = {} # key for ids value for labels
        for video in videos:
            video_id = video.split('/')[-1]
            self.meta_data[video_id] = {}

            all_tubes = glob.glob(os.path.join(video, '/local/*/*/'))
            for tube in all_tubes:
                tube_id = tube.split('/')[-1]
                self.meta_tube[tube_id] = {
                    'frame_id': [],
                    'bbox_pos': [],
                    'agent_id': [],
                    'action_id': [],
                    'loc_id': []
                }

                self.meta_data[video_id][tube] = {}

                labels = open(os.path.join(tube, '/label.csv'), 'r')
                labels = labels.readlines()

                for label in labels:
                    frame_id, x1, y1, x2, y2, agent_id, action_id, \
                        loc_id, _ = label.split(',')
                    
                    self.meta_data[video_id][tube]['global_id'] = frame_id + '.jpg'
                    self.meta_data[video_id][tube]['bbox_pos'] = list(map(float, [x1, y1, x2, y2]))
                    self.meta_data[video_id][tube]['agent_id'] = agent_id
                    self.meta_data[video_id][tube]['action_id'] = action_id
                    self.meta_data[video_id][tube]['loc_id'] = loc_id

                    self.meta_tube[tube_id]['frame_id'].append(frame_id + '.jpg')
                    self.meta_tube[tube_id]['bbox_pos'].append(list(map(float, [x1, y1, x2, y2])))
                    self.meta_tube[tube_id]['agent_id'].append(agent_id)
                    self.meta_tube[tube_id]['action_id'].append(action_id)
                    self.meta_tube[tube_id]['loc_id'].append(loc_id)
        
        self.video_ids = self.meta_data.keys()

        self.all_datas = self.combine_datas(self.meta_tube)

    
    def getitem(self, idx):
    
    def len(self):
        cont = 0
        for key in self.meta_keys:


    def combine_datas(self, meta_tube):
        frame_queue = queue()
"""
agent_labels = ['Ped', 'Car', 'Cyc', 'Mobike', 'SmalVeh', 'MedVeh', 'LarVeh', 'Bus', 'EmVeh', 'TL']
action_labels = ['Red', 'Amber', 'Green', 'MovAway', 'MovTow', 'Mov', 'Rev', 'Brake', 'Stop', 'IncatLft', 'IncatRht', 'HazLit', 'TurLft', 'TurRht', 'MovRht', 'MovLft', 'Ovtak', 'Wait2X', 'XingFmLft', 'XingFmRht', 'Xing', 'PushObj']
loc_labels = ['VehLane', 'OutgoLane', 'OutgoCycLane', 'OutgoBusLane', 'IncomLane', 'IncomCycLane', 'IncomBusLane', 'Pav', 'LftPav', 'RhtPav', 'Jun', 'xing', 'BusStop', 'parking', 'LftParking', 'rightParking']

action_order = [i for i in range(len(action_labels))]
loc_order = [i for i in range(len(loc_labels))]

class track2_dataset(nn.Module):
    def __init__(self, args) -> None:
        self.datapath = args.dataset_path
        self.window_size = int(args.window_size)
        self.shape = list(map(int, args.input_shape))

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        videos = glob.glob(self.datapath + "/*/")

        self.meta_tube = {} # key for ids value for labels
        for video in videos:
            video_id = video.split('/')[-2]

            all_tubes = glob.glob(video + 'local/*/*/')
            for tube in all_tubes:
                tube_id = tube.split('/')[-2]
                self.meta_tube[tube_id] = {
                    'video_id': video_id,
                    'frame_id': [],
                    'bbox_pos': [],
                    'agent_id': [],
                    'action_id': [],
                    'loc_id': []
                }

                labels = open(tube + 'label.csv', 'r')
                labels = labels.readlines()

                for label in labels:
                    frame_id, x1, y1, x2, y2, agent_id, action_id, \
                        loc_id, _ = label.split(',')

                    self.meta_tube[tube_id]['frame_id'].append(frame_id + '.jpg')
                    self.meta_tube[tube_id]['bbox_pos'].append(list(map(float, [x1, y1, x2, y2])))
                    self.meta_tube[tube_id]['agent_id'].append(agent_id)
                    self.meta_tube[tube_id]['action_id'].append(action_id)
                    self.meta_tube[tube_id]['loc_id'].append(loc_id)
        
        # self.video_ids = self.meta_data.keys()
        self.all_datas = self.combine_datas(self.meta_tube)

    
    def __getitem__(self, idx):
        window = self.all_datas[idx]

        datas = {'stacked_img': [], 'label': []}

        # stacked_img = None

        for frame in window:
            video_id = frame['video_id']
            tube_id = frame['tube_id']
            local_frame_id = frame['local_frame_id']
            global_frame_id = frame['global_frame_id']
            bbox_pos = frame['bbox_pos']
            action_id = frame['action_id']
            agent_id = frame['agent_id']
            loc_id = frame['loc_id']
            
            local_img = cv2.imread(os.path.join(self.datapath, video_id, 'local', 
                str(agent_id) + '_' + agent_labels[int(agent_id)], tube_id, local_frame_id))
            global_img = cv2.imread(os.path.join(
                self.datapath, video_id, 'global', global_frame_id))
            
            local_img = cv2.resize(cv2.cvtColor(local_img, cv2.COLOR_BGR2RGB), self.shape)
            global_img = cv2.resize(cv2.cvtColor(global_img, cv2.COLOR_BGR2RGB), self.shape)

            stack_img = np.concatenate((global_img, local_img), axis=-1)
            
            label_dict = {
                'data_anno': {
                    'video_id': video_id,
                    'tube_id': tube_id,
                    'frame_id': local_frame_id,
                    'agent_id': agent_id + '_{}'.format(agent_labels[int(agent_id)]),
                    'action_id': action_id + '_{}'.format(action_labels[int(action_id)]),
                    'loc_id': loc_id + '_{}'.format(loc_labels[int(loc_id)])
                },
                'bbox_pos': bbox_pos,
                'action_label': [0 for _ in range(len(action_labels))],
                'loc_label': [0 for _ in range(len(loc_labels))]
            }

            label_dict['action_label'][action_order.index(int(action_id))] = 1
            label_dict['loc_label'][loc_order.index(int(loc_id))] = 1
            label_dict['action_label'] = torch.FloatTensor(label_dict['action_label'])
            label_dict['loc_label'] = torch.FloatTensor(label_dict['loc_label'])

            datas['stacked_img'].append(stack_img)
            datas['label'].append(label_dict)

        data = np.concatenate([img for img in datas['stacked_img']], axis=-1)
        data = self.transform(data)
        label = datas['label']

        return data, label

    
    def __len__(self):
        return len(self.all_datas)


    def combine_datas(self, meta_tube):
        all_datas = []

        for key in meta_tube.keys():
            tubes = meta_tube[key]

            window_queue = deque(maxlen=self.window_size)
            for i in range(len(tubes['frame_id'])):
                window_queue.append({
                    'video_id': tubes['video_id'],
                    'tube_id': key,
                    'local_frame_id': tubes['frame_id'][i],
                    'global_frame_id': tubes['frame_id'][i],
                    'bbox_pos': tubes['bbox_pos'][i],
                    'agent_id': tubes['agent_id'][i],
                    'action_id': tubes['action_id'][i],
                    'loc_id': tubes['loc_id'][i]
                })

                if len(window_queue) == self.window_size:
                    all_datas.append(window_queue.copy())
            
        return all_datas
        


class Tracklet_Dataset(nn.Module):
    def __init__(self, tracklet, windows_size):
        self.windows = []
        windows_deque = deque(maxlen=windows_size)

        for t in tracklet:
            windows_deque.append(t)

            if len(windows_deque) == windows_size:
                stacked_img = np.concatenate(windows_deque, axis=-1)
                self.windows.append(stacked_img)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return transforms.ToTensor(self.windows[idx])


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-datapath", "-d", default="/datasets/roadpp/Track2/")
    parser.add_argument('--window_size', '-wsize',  default=4, help='path of dataset')
    parser.add_argument('--input_shape', '-inshape', nargs='+', default=(480, 720), help='path of dataset')
    args = parser.parse_args()
    
    # 3:2 1920*1280
    dataset = track2_dataset(args)

    for idx in tqdm(range(len(dataset))):
        dataset[idx]
        pass
