import os
import cv2
import glob
import pickle
import numpy as np
from ultralytics import YOLO
from Track2.dataset import Tracklet_Dataset
from utils.linear_interpolation import tube_interpolation
from utils.tube_processing import tube_change_axis


def out_of_range(x, y, max_x, max_y):
    x = min(max(x, 0), max_x)
    y = min(max(y, 0), max_y)
    return x, y


def make_tube(tube, video_name, tracker, orig_shape, t2_shape, submit_shape):
    """
    Make submit tube using track algorithm.
    
    Args:
        tube (dict): Final submit data(See Submission_format.py)
        video_name (str): video name, ex: val_00001, train_00001...etc
        tracker (object): Yolov8's track result.
        orig_shape (tuple): video's shape.
        t2_shape (tuple): track2 input shape.
        submit_shape (tuple): final submit shape.
    
    tube:
        tube['agent']: {
            'label_id': class index, 
            'scores': bounding box scores, 
            'boxes': bounding box coordinates (absolute), 
            'score': tube score(we using np.mean(scores)), 
            'frames': frames across which the tube spans
            }

        tube['event']: {
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
    for t in tracker:
        frame_num += 1
        if t.boxes.is_track:
            frame_img = t.orig_img
            global_img = cv2.resize(cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB), t2_shape)

            for b in t.boxes.data:
                x1, y1, x2, y2, track_id, conf, label_id = b
                
                # Convert tensor values to Python scalars
                x1, y1, x2, y2, track_id, conf, label_id = (
                    x1.item(), y1.item(), x2.item(), y2.item(),
                    int(track_id.item()), conf.item(), int(label_id.item())
                )

                x1, y1 = out_of_range(x1, y1, t.orig_shape[1], t.orig_shape[0])
                x2, y2 = out_of_range(x2, y2, t.orig_shape[1], t.orig_shape[0])

                local_img = frame_img[int(y1) : int(y2), int(x1) : int(x2)]
                local_img = cv2.resize(cv2.cvtColor(local_img, cv2.COLOR_BGR2RGB), t2_shape)
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
                    stack_imgs[track_id] = [stack_img]
                else:
                    # agent
                    tracklet[track_id]['scores'] = np.append(tracklet[track_id]['scores'], conf)
                    tracklet[track_id]['boxes'] = np.append(tracklet[track_id]['boxes'], [[x1, y1, x2, y2]], axis=0)
                    tracklet[track_id]['frames'] = np.append(tracklet[track_id]['frames'], frame_num)

                    # event
                    stack_imgs[track_id].append(stack_img)

    agent_list = []
    event_list = []
    
    for tube_id, tube_data in tracklet.items():
        # agent
        tube_data = tube_interpolation(tube_data)
        tube_data = tube_change_axis(tube_data, orig_shape, submit_shape) # change axis to submit_shape
        tube_data['score'] = np.mean(tube_data['scores'])
        agent_list.append(tube_data)

        # event
        tube_data['stack_imgs'] = stack_imgs[tube_id]
        event_list.append(tube_data)
    
    tube['agent'][video_name] = agent_list
    tube['event'][video_name] = event_list

    return 0


def track2(tube, windows_size):
    for t in tube:
        dataset = Tracklet_Dataset(t['stack_imgs'], windows_size)

    return 0


def main(model, video_path, imgsz, devices, pkl_name, submit_shape, save_res):

    tube = {
        'agent': {},
        'event': {}
    }

    for v in sorted(glob.glob(os.path.join(video_path, '*.mp4'))):
        video_name = v.split('/')[-1].split('.')[0]
        
        # tracking Using BoT-SORT
        tracker = model.track(
            source=v,
            imgsz=imgsz,
            device=devices,
            stream=True,
            conf = 0.0
        )
        
        make_tube(tube, video_name, tracker, imgsz, t2_input_shape, submit_shape)

        # with open(pkl_name, 'wb') as f:
        #     pickle.dump(tube, f)

        track2(tube['event'][video_name], windows_size)


    if save_res:
        if os.path.exists(pkl_name):
            os.remove(pkl_name)

        with open(pkl_name, 'wb') as f:
            pickle.dump(tube, f)


if __name__ == '__main__':
    video_path = '/mnt/datasets/roadpp/videos'
    model_path = '/home/Ricky/0_Project/ROADpp_challenge_ICCV2023/runs/detect/yolov8l_T1_1280_batch_8_/weights/best.pt'
    devices = '0'
    imgsz = [1280, 1280]
    submit_shape = (600, 840)
    t2_input_shape = (240, 360)
    windows_size = 4
    yolo_1280 = YOLO(model_path)
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # video = cv2.VideoWriter('test.mp4', fourcc, 5, (1920, 1280))
    
    main(yolo_1280, video_path, imgsz, devices, 'T1_train.pkl', submit_shape, save_res=False)
