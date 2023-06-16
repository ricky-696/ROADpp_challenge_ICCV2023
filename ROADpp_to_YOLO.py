import os
import json
import cv2
import glob
import shutil


def debug_draw(img, b, filename, w, h):
    x1, y1, x2, y2 = b
    x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
    cv2.rectangle(img, (x1, y1), (x2, y2),(0, 0, 255), 3)
    cv2.imwrite(filename, img)


def bbox_to_yolo(_class, x1, y1, x2, y2):
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    return str(_class) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(w) + ' ' + str(h)


def img_to_yolo(video_folder, train_folder):
    # 取得所有影片資料夾的名稱
    video_folders = sorted(os.listdir(video_folder))

    # 迭代每個影片資料夾
    for video_folder_name in video_folders:
        video_path = os.path.join(video_folder, video_folder_name)
        
        # 檢查路徑是否為資料夾
        if os.path.isdir(video_path):
            # 取得影片名稱
            video_name = video_folder_name
            
            # 取得影像檔案清單
            image_files = os.listdir(video_path)
            
            # 迭代每個影像檔案
            for image_file in image_files:
                image_path = os.path.join(video_path, image_file)
                
                # 檢查檔案是否為影像
                if os.path.isfile(image_path) and image_file.endswith(".jpg"):
                    # 取得frame_id
                    frame_id = image_file.split(".")[0]
                    
                    # 設定目標檔案路徑
                    target_path = os.path.join(train_folder, f"{video_name}_{frame_id}.jpg")
                    
                    # 複製影像檔案到目標資料夾並重新命名
                    shutil.copy(image_path, target_path)
                    print(f"copy {image_path} to {target_path}")


def gt_to_yolo(gt_file):
    # save format: (video_name)_(frame_id).txt
    # example:      train_000000_00001.txt
    save_path = os.path.join(train_folder, 'labels')

    print('Loading json file...')
    with open(gt_file, 'r') as f:
        gt_dict = json.load(f)

    for video_name, video in gt_dict['db'].items():
        
        for frame_id, data in video['frames'].items():
            file_name = str(video_name) + '_' + str(frame_id).zfill(5)
            img_width, img_height = data['width'], data['height']

            # img = cv2.imread(os.path.join(train_folder, file_name + '.jpg'))

            save_file = os.path.join(save_path, file_name + '.txt')
            print(f'writing {save_file}')

            # bbox format: x1, y1, x2, y2
            if 'annos' in data:
                for box_id, annos in data['annos'].items():
                    agent_label = annos['agent_ids'][0]
                    x1, y1, x2, y2 = annos['box']
                    yolo_bbox_label = bbox_to_yolo(agent_label, x1, y1, x2, y2)

                    if not os.path.exists(save_file):
                        with open(save_file, 'w') as f:
                            f.write(yolo_bbox_label)
                    else:
                        with open(save_file, 'a') as f:
                            f.write('\n' + yolo_bbox_label)
                
                
def move_img():
    source_folder = '/mnt/datasets/roadpp/train'
    target_folder = '/mnt/datasets/roadpp/train/images'

    # 确保目标文件夹存在
    os.makedirs(target_folder, exist_ok=True)

    # 获取原始文件夹中的所有文件
    file_list = os.listdir(source_folder)

    # 遍历文件列表并移动文件
    for file_name in file_list:
        if file_name != 'images':
            source_file = os.path.join(source_folder, file_name)
            target_file = os.path.join(target_folder, file_name)
            shutil.move(source_file, target_file)
            print(f'move {source_file} to {target_file}')


if __name__ == '__main__':
    img_folder = '/mnt/datasets/roadpp/rgb-images'
    train_folder = '/mnt/datasets/roadpp/train'
    val_floder = '/mnt/datasets/roadpp/valid'
    test_floder = '/mnt/datasets/roadpp/test'
    gt_file = '/mnt/datasets/roadpp/road_waymo_trainval_v1.0.json'

    # move_img()
    # img_to_yolo(img_folder, os.path.join(train_folder, 'images'))
    gt_to_yolo(gt_file)

