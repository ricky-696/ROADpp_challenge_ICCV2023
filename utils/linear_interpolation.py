import pickle
import numpy as np
from tqdm import tqdm


def tube_interpolation(tube):
    frames = tube['frames']
    scores = tube['scores']
    boxes = tube['boxes']
    
    interpolated_frames = np.arange(frames[0], frames[-1] + 1)  
    interpolated_scores = np.interp(interpolated_frames, frames, scores)  
    interpolated_boxes = np.empty((len(interpolated_frames), 4))  
    
    for i, axis in enumerate([0, 1, 2, 3]):
        interpolated_boxes[:, i] = np.interp(interpolated_frames, frames, boxes[:, axis])
    
    interpolated_tube = {
        'label_id': tube['label_id'],
        'scores': interpolated_scores,
        'boxes': interpolated_boxes,
        'score': tube['score'],
        'frames': interpolated_frames
    }
    
    return interpolated_tube


def debug():
    tube_a = {
        'label_id': 0,
        'scores': np.array([0.11491841, 0.16743928, 0.25436875, 0.3424321, 0.13464102, 0.5]),
        'boxes': np.array([[473.2007, 229.3157, 485.1342, 255.87836],
                            [475.29144, 229.99019, 487.3297, 256.3758],
                            [476.16983, 232.67128, 487.9435, 256.62274],
                            [477.0421, 233.207, 489.6095, 261.4667],
                            [476.92032, 231.34416, 489.44467, 258.55612],
                            [476.92032, 231.34416, 489.44467, 258.55612]]),
        'score': 0.2984004318714142,
        'frames': np.array([3, 4, 6, 7, 8, 10])
    }

    interpolated_tube_a = tube_interpolation(tube_a)
    print("Interpolated Tube A:", interpolated_tube_a)


def pkl_tube_interpolation(file_path, label_type, pkl_name):
    with open(file_path, 'rb') as f:
        tubes = pickle.load(f)
        
    for video_name in tqdm(tubes[label_type], desc='Videos'):
        for i, t in enumerate(tubes[label_type][video_name]):
            tubes[label_type][video_name][i] = tube_interpolation(tubes[label_type][video_name][i])
            
    check_missing_frames(tubes)
    
    with open(pkl_name, 'wb') as f:
        pickle.dump(tubes, f)
            

def check_missing_frames(tubes):
    for label_type in tubes.keys():
        for video_name, v_tube in tubes[label_type].items():
            for i, t in enumerate(v_tube):
                frames = t['frames']
                missing_frames = []

                for j in range(frames[0], frames[-1] + 1):
                    if j not in frames:
                        missing_frames.append(j)

                if missing_frames:
                    print(f"Missing frames in {label_type} - {video_name}, Tube {i}:", missing_frames)
    

if __name__ == '__main__':
    # debug()
    file_path = '/home/Ricky/0_Project/ROADpp_challenge_ICCV2023/T1_submit_0.pkl'
    
    pkl_tube_interpolation(
        file_path,
        'agent',
        'T1_interpolation.pkl'
        )
    