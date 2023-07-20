import os
import cv2
import csv
import json
import glob
import random
import shutil

if __name__ == '__main__':
    # create local img(cut bbox img in every frame)
    agent_labels = ['Ped', 'Car', 'Cyc', 'Mobike', 'SmalVeh', 'MedVeh', 'LarVeh', 'Bus', 'EmVeh', 'TL']
    img_folder = '/datasets/roadpp/Track2'
    gt_file = '/datasets/roadpp/road_waymo_trainval_v1.0.json'
    incomplete_labels = 0


    print('Loading json file...')
    with open(gt_file, 'r') as f:
        gt_dict = json.load(f)

    annos = gt_dict['db']['train_00407']['agent_tubes']['annos']
    gt_dict['db']['train_00407']['agent_tubes']['283ae350-0e37-460c-9d1a-d5d62a277a61-001']
    gt_dict['db']['train_00407']['frames']['1']['annos']['b_1']
    print(annos)