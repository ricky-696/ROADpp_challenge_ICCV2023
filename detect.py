import os
import cv2
import glob
import torch
import pickle
import numpy as np
from ultralytics import YOLO
from Track2.dataset import Tracklet_Dataset

from utils.opt import arg_parse
from utils.linear_interpolation import tube_interpolation
from utils.tube_processing import tube_change_axis

import sys
sys.path.append('./Track2')


def out_of_range(x, y, max_x, max_y):
    x = min(max(x, 0), max_x)
    y = min(max(y, 0), max_y)
    return x, y


def make_tube(args):
    """
    Make submit tube using track algorithm.
    
    Args:
        tube (dict): Final submit data(See Submission_format.py)
        video_name (str): video name, ex: val_00001, train_00001...etc
        tracker (object): Yolov8's track result.
        video_shape (tuple): video's shape.
        t2_input_shape (tuple): track2 input shape.
        submit_shape (tuple): final submit shape.
    
    tube:
        tube['agent']['video_name'][idx]: {
            'label_id': class index, 
            'scores': bounding box scores, 
            'boxes': bounding box coordinates (absolute), 
            'score': tube score(we using np.mean(scores)), 
            'frames': frames across which the tube spans
        }

        tube['event']['video_name'][idx]: {
            'label_id': class index, 
            'scores': bounding box scores, 
            'boxes': bounding box coordinates (absolute), 
            'score': tube score(we using np.mean(scores)), 
            'frames': frames across which the tube spans
            'stack_imgs': concated global & local img by frames
        }
    """
    
    tracklet = {}
    stack_imgs = {} 
    frame_num = 0
    
    # Tracker.boxes.data(Tensor): x1, y1, x2, y2, track_id, conf, label_id
    for t in args.tracker:
        frame_num += 1
        if t.boxes.is_track:
            frame_img = t.orig_img
            global_img = cv2.resize(cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB), args.t2_input_shape)

            for b in t.boxes.data:
                x1, y1, x2, y2, track_id, conf, label_id = b
                
                # Convert tensor values to Python scalars
                x1, y1, x2, y2, track_id, conf, label_id = (
                    x1.item(), y1.item(), x2.item(), y2.item(),
                    int(track_id.item()), conf.item(), int(label_id.item())
                )

                x1, y1 = out_of_range(x1, y1, t.orig_shape[1], t.orig_shape[0])
                x2, y2 = out_of_range(x2, y2, t.orig_shape[1], t.orig_shape[0])

                if args.mode == 'Track2':
                    local_img = frame_img[int(y1) : int(y2), int(x1) : int(x2)]
                    local_img = cv2.resize(cv2.cvtColor(local_img, cv2.COLOR_BGR2RGB), args.t2_input_shape)
                    stack_img = np.concatenate((global_img, local_img), axis=-1)
                
                if track_id not in tracklet:
                    # agent
                    tracklet[track_id] = {
                        'label_id': label_id,
                        'scores': np.array([conf]),
                        'boxes': np.array([[x1, y1, x2, y2]]),
                        'score': 0.0,
                        'frames': np.array([frame_num])
                    }

                    # event
                    if args.mode == 'Track2':
                        stack_imgs[track_id] = [stack_img]
                else:
                    # agent
                    tracklet[track_id]['scores'] = np.append(tracklet[track_id]['scores'], conf)
                    tracklet[track_id]['boxes'] = np.append(tracklet[track_id]['boxes'], [[x1, y1, x2, y2]], axis=0)
                    tracklet[track_id]['frames'] = np.append(tracklet[track_id]['frames'], frame_num)

                    # event
                    if args.mode == 'Track2':
                        stack_imgs[track_id].append(stack_img)

    agent_list = []
    event_list = []
    
    for tube_id, tube_data in tracklet.items():
        # agent
        tube_data = tube_interpolation(tube_data)
        tube_data = tube_change_axis(tube_data, args.video_shape, args.submit_shape) # change axis to submit_shape
        tube_data['score'] = np.mean(tube_data['scores'])
        agent_list.append(tube_data.copy())

        # event
        if args.mode == 'Track2':
            tube_data['stack_imgs'] = stack_imgs[tube_id]
            event_list.append(tube_data)
    
    args.tube['agent'][args.video_name] = agent_list

    if args.mode == 'Track2':
        args.tube['event'][args.video_name] = event_list

    return 0


def make_t2_tube(tube, action_cls, loc_cls):


    return tube


# ToDo: after predict one videos, need to release event's stacked img, else OOM
def track2(args):
    event_tubes_list = []

    with torch.no_grad():
        for video_id, tubes in args.tube['event'].items():
            for t in tubes:
                # Create a dataset using Sliding Windows.
                action_dataset = Tracklet_Dataset(
                    mode='action',
                    tracklet=t['stack_imgs'], 
                    windows_size=args.windows_size
                )

                loc_dataset = Tracklet_Dataset(
                    mode='loc',
                    tracklet=t['stack_imgs'], 
                    windows_size=args.windows_size
                )

                # predict
                action_cls = []
                for tracklet in action_dataset:
                    input = torch.unsqueeze(tracklet, 0)
                    pred = args.action_detector(input)
                    cls = torch.argmax(pred, dim=1)
                    action_cls.append(cls.item())

                loc_cls = []
                for stack_img in loc_dataset:
                    input = torch.unsqueeze(stack_img, 0)
                    pred = args.action_detector(input)
                    cls = torch.argmax(pred, dim=1)
                    loc_cls.append(cls.item())

                # Padding and Matching
                event_tubes_list.append(make_t2_tube(t, action_cls, loc_cls))

    return 0


def main(args):
    """
        Args: see utils/opt.py
    """

    if args.mode == 'Track2':
        args.tube = {
            'agent': {},
            'event': {}
        }
    else:
        args.tube = {
            'agent': {}
        }

    for v in sorted(glob.glob(os.path.join(args.video_path, '*.mp4'))):
        args.video_name = v.split('/')[-1].split('.')[0]
        
        # tracking Using BoT-SORT
        args.tracker = args.yolo.track(
            source=v,
            imgsz=args.imgsz,
            device=args.devices,
            stream=True,
            conf = 0.0
        )
        
        make_tube(args)

        # memory_size = sys.getsizeof(args.tube)

        # # debug for one video
        # with open(pkl_name, 'wb') as f:
        #     pickle.dump(tube, f)

        if args.mode == 'Track2':
            track2(args)


    if args.save_res:
        if os.path.exists(args.pkl_name):
            os.remove(args.pkl_name)

        with open(args.pkl_name, 'wb') as f:
            pickle.dump(args.tube, f)


if __name__ == '__main__':
    args = arg_parse()
    assert args.mode == 'Track1' or args.mode == 'Track2', 'detect mode only accept "Track1" or "Track2".'
    args.yolo = YOLO(args.yolo_path)

    # debug_args
    # args.devices = '1'
    args.mode = 'Track2'
    
    if args.mode == 'Track2':
        args.action_detector = torch.load(args.action_detector_path)
        args.action_detector.eval()
    
    main(args)
