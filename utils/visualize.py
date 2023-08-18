import os
import cv2
import glob
import json
import pickle

import sys
sys.path.append('../ROADpp_challenge_ICCV2023')
sys.path.append('../ROADpp_challenge_ICCV2023/ROAD_Waymo_Baseline')

from tube_processing import bbox_normalized, norm_box_into_absolute
from ROAD_Waymo_Baseline.modules.evaluation import get_gt_tubes


def plot(frame, box, text, color=(0, 255, 0)):
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    # Draw the bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Add text next to the bounding box
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (x1, y1 - 10)  # Adjust the text position based on your preference
    font_scale = 1.0
    color = color
    thickness = 1
    cv2.putText(frame, text, org, font, font_scale, color, thickness, cv2.LINE_AA)
    
    return frame


def plot_pred():
    for tubes in detections['triplet'][video_name]:
        for i in range(len(tubes['frames'])):
            frame_num = tubes['frames'][i] - 1
            cls = used_labels['triplet_labels'][tubes['label_id']]
            box = norm_box_into_absolute(
                bbox=bbox_normalized(tubes['boxes'][i], 840, 600),
                img_w=1920,
                img_h=1280
            )
            
            ori_frames[frame_num] = plot(ori_frames[frame_num], box, cls)
            
    output_video_path = 'train_pred.mp4'
    height, width, layers = ori_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))
    
    # Write the frames into the output video
    for frame in ori_frames:
        out.write(frame)
    
    out.release()
    cv2.destroyAllWindows()


def plot_gt():
    for tubes in gt_tubes[video_name]:
        for i in range(len(tubes['frames'])):
            frame_num = tubes['frames'][i] - 1
            cls = used_labels['triplet_labels'][tubes['label_id']]
            box = norm_box_into_absolute(
                bbox=bbox_normalized(tubes['boxes'][i], 840, 600),
                img_w=1920,
                img_h=1280
            )
            
            ori_frames[frame_num] = plot(ori_frames[frame_num], box, cls, (0, 0, 255))
            
    output_video_path = 'train_gt.mp4'
    height, width, layers = ori_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))
    
    # Write the frames into the output video
    for frame in ori_frames:
        out.write(frame)
    
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    used_labels = {"agent_labels": ["Ped", "Car", "Cyc", "Mobike", "SmalVeh", "MedVeh", "LarVeh", "Bus", "EmVeh", "TL"],
                   "action_labels": ["Red", "Amber", "Green", "MovAway", "MovTow", "Mov", "Rev", "Brake", "Stop", "IncatLft", "IncatRht", "HazLit", "TurLft", "TurRht", "MovRht", "MovLft", "Ovtak", "Wait2X", "XingFmLft", "XingFmRht", "Xing", "PushObj"],
                   "loc_labels": ["VehLane", "OutgoLane", "OutgoCycLane", "OutgoBusLane", "IncomLane", "IncomCycLane", "IncomBusLane", "Pav", "LftPav", "RhtPav", "Jun", "xing", "BusStop", "parking", "LftParking", "rightParking"],
                   "duplex_labels": ["Ped-MovAway", "Ped-MovTow", "Ped-Mov", "Ped-Stop", "Ped-Wait2X", "Ped-XingFmLft", "Ped-XingFmRht", "Ped-Xing", "Ped-PushObj", "Car-MovAway", "Car-MovTow", "Car-Brake", "Car-Stop", "Car-IncatLft", "Car-IncatRht", "Car-HazLit", "Car-TurLft", "Car-TurRht", "Car-MovRht", "Car-MovLft", "Car-XingFmLft", "Car-XingFmRht", "Cyc-MovAway", "Cyc-MovTow", "Cyc-Stop", "Mobike-Stop", "MedVeh-MovAway", "MedVeh-MovTow", "MedVeh-Brake", "MedVeh-Stop", "MedVeh-IncatLft", "MedVeh-IncatRht", "MedVeh-HazLit", "MedVeh-TurRht", "MedVeh-XingFmLft", "MedVeh-XingFmRht", "LarVeh-MovAway", "LarVeh-MovTow", "LarVeh-Stop", "LarVeh-HazLit", "Bus-MovAway", "Bus-MovTow", "Bus-Brake", "Bus-Stop", "Bus-HazLit", "EmVeh-Stop", "TL-Red", "TL-Amber", "TL-Green"], 
                   "triplet_labels": ["Ped-MovAway-LftPav", "Ped-MovAway-RhtPav", "Ped-MovAway-Jun", "Ped-MovTow-LftPav", "Ped-MovTow-RhtPav", "Ped-MovTow-Jun", "Ped-Mov-OutgoLane", "Ped-Mov-Pav", "Ped-Mov-RhtPav", "Ped-Stop-OutgoLane", "Ped-Stop-Pav", "Ped-Stop-LftPav", "Ped-Stop-RhtPav", "Ped-Stop-BusStop", "Ped-Wait2X-RhtPav", "Ped-Wait2X-Jun", "Ped-XingFmLft-Jun", "Ped-XingFmRht-Jun", "Ped-XingFmRht-xing", "Ped-Xing-Jun", "Ped-PushObj-LftPav", "Ped-PushObj-RhtPav", "Car-MovAway-VehLane", "Car-MovAway-OutgoLane", "Car-MovAway-Jun", "Car-MovTow-VehLane", "Car-MovTow-IncomLane", "Car-MovTow-Jun", "Car-Brake-VehLane", "Car-Brake-OutgoLane", "Car-Brake-Jun", "Car-Stop-VehLane", "Car-Stop-OutgoLane", "Car-Stop-IncomLane", "Car-Stop-Jun", "Car-Stop-parking", "Car-IncatLft-VehLane", "Car-IncatLft-OutgoLane", "Car-IncatLft-IncomLane", "Car-IncatLft-Jun", "Car-IncatRht-VehLane", "Car-IncatRht-OutgoLane", "Car-IncatRht-IncomLane", "Car-IncatRht-Jun", "Car-HazLit-IncomLane", "Car-TurLft-VehLane", "Car-TurLft-Jun", "Car-TurRht-Jun", "Car-MovRht-OutgoLane", "Car-MovLft-VehLane", "Car-MovLft-OutgoLane", "Car-XingFmLft-Jun", "Car-XingFmRht-Jun", "Cyc-MovAway-OutgoCycLane", "Cyc-MovAway-RhtPav", "Cyc-MovTow-IncomLane", "Cyc-MovTow-RhtPav", "MedVeh-MovAway-VehLane", "MedVeh-MovAway-OutgoLane", "MedVeh-MovAway-Jun", "MedVeh-MovTow-IncomLane", "MedVeh-MovTow-Jun", "MedVeh-Brake-VehLane", "MedVeh-Brake-OutgoLane", "MedVeh-Brake-Jun", "MedVeh-Stop-VehLane", "MedVeh-Stop-OutgoLane", "MedVeh-Stop-IncomLane", "MedVeh-Stop-Jun", "MedVeh-Stop-parking", "MedVeh-IncatLft-IncomLane", "MedVeh-IncatRht-Jun", "MedVeh-TurRht-Jun", "MedVeh-XingFmLft-Jun", "MedVeh-XingFmRht-Jun", "LarVeh-MovAway-VehLane", "LarVeh-MovTow-IncomLane", "LarVeh-Stop-VehLane", "LarVeh-Stop-Jun", "Bus-MovAway-OutgoLane", "Bus-MovTow-IncomLane", "Bus-Stop-VehLane", "Bus-Stop-OutgoLane", "Bus-Stop-IncomLane", "Bus-Stop-Jun", "Bus-HazLit-OutgoLane"]}

    video_frames_path = '/mnt/datasets/roadpp/rgb-images'
    video_name = 'train_00000'
    frames_path = os.path.join(video_frames_path, video_name, '*.jpg')
    frames_path = glob.glob(frames_path)
    
    # read frames
    ori_frames = []
    for path in sorted(
        frames_path, 
        key=lambda x: int(x.split('/')[-1].split('.')[0])
    ):
        ori_frames.append(cv2.imread(path))
        
    # read gt_file
    gt_file = '/mnt/datasets/roadpp/road_waymo_trainval_v1.0.json'
    with open(gt_file, 'r') as f:
        final_annots = json.load(f)
        
    video_list, gt_tubes = get_gt_tubes(
        final_annots, 
        subset='train', 
        label_type='triplet', 
        dataset='triplet'
        )
    
    # read pred
    det_file = '/home/Ricky/0_Project/ROADpp_challenge_ICCV2023/T2_train.pkl'
    with open(det_file, 'rb') as fff:
        detections = pickle.load(fff)
        
    # plot_pred()
    plot_gt()
    
            
        
    
    
    