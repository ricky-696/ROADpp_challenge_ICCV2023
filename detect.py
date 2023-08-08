import os
import cv2
import glob
import pickle
import numpy as np
from ultralytics import YOLO


def make_tube(tube, video_name, tracker):
    """
    Make submit tube using track algorithm.
    
    Args:
        tube (dict): Final submit data(See Submission_format.py)
        video_name (str): video name, ex: val_00001, train_00001...etc
        tracker (object): Yolov8's track result.
        
    tracklet[track_id] = {
        'label_id': class index, 
        'scores': bounding box scores, 
        'boxes': bounding box coordinates (absolute), 
        'score': tube score(we using np.mean(scores)), 
        'frames': frames across which the tube spans
        }
    """
    
    tracklet = {} 
    frame_num = 0
    
    # Tracker.boxes.data(Tensor): x1, y1, x2, y2, track_id, conf, label_id
    for t in tracker:
        frame_num += 1
        if t.boxes.is_track:
            for b in t.boxes.data:
                x1, y1, x2, y2, track_id, conf, label_id = b
                
                # Convert tensor values to Python scalars
                x1, y1, x2, y2, track_id, conf, label_id = (
                    x1.item(), y1.item(), x2.item(), y2.item(),
                    int(track_id.item()), conf.item(), int(label_id.item())
                )
                
                if track_id not in tracklet:
                    # Create the tube entry if it doesn't exist in the tracklet dictionary
                    tracklet[track_id] = {
                        'label_id': label_id,
                        'scores': np.array([conf]),
                        'boxes': np.array([[x1, y1, x2, y2]]),
                        'score': 0.0,
                        'frames': np.array([frame_num])
                    }
                else:
                    # Add the data to the existing tube entry
                    tracklet[track_id]['scores'] = np.append(tracklet[track_id]['scores'], conf)
                    tracklet[track_id]['boxes'] = np.append(tracklet[track_id]['boxes'], [[x1, y1, x2, y2]], axis=0)
                    tracklet[track_id]['frames'] = np.append(tracklet[track_id]['frames'], frame_num)
    tracklet_list = []
    for tube_id, tube_data in tracklet.items():
        tube_data['score'] = np.mean(tube_data['scores'])
        tracklet_list.append(tube_data)
    
    tube['agent'][video_name] = tracklet_list
    
    # return tracklet
        

def make_tracklet(tracker):
    # for every track_id, build tracklet = [frame1_bbox_cut_img, frame2_bbox_cut_img, ...]
    tracklet = {}
    return tracklet


def track1(model, video_path, imgsz, devices, pkl_name):
    if os.path.exists(pkl_name):
        os.remove(pkl_name)

    tube = {
        'agent': {}
    }

    for v in sorted(glob.glob(os.path.join(video_path, '*.mp4'))):
        video_name = v.split('/')[-1].split('.')[0]
        
        tracker = model.track(
            source=v,
            imgsz=imgsz,
            device=devices,
            stream=True,
            conf = 0.0
        )
        
        make_tube(tube, video_name, tracker)

    with open(pkl_name, 'wb') as f:
        pickle.dump(tube, f)


if __name__ == '__main__':
    video_path = '/datasets/roadpp/test_videos'
    model_path = '/home/Ricky/0_Project/ROADpp_challenge_ICCV2023/runs/detect/yolov8l_T1_1280_batch_8_/weights/best.pt'
    devices = '0'
    imgsz = [1280, 1280]
    yolo_1280 = YOLO(model_path)
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # video = cv2.VideoWriter('test.mp4', 
    #                         fourcc, 5, (1920, 1280))
    
    track1(yolo_1280, video_path, imgsz, devices, 'T1_submit_conf_0.pkl')
