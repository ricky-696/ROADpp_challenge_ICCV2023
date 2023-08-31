import os
import re
import cv2
import glob
import json
import random
import shutil
from tqdm import tqdm


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
                
                
def move_img_to_train_img():
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


def cut_train_valid(train_folder, val_folder, mode='video', ratio=0.9):
    train_images_folder = os.path.join(train_folder, 'images')
    train_labels_folder = os.path.join(train_folder, 'labels')
    valid_images_folder = os.path.join(val_folder, 'images')
    valid_labels_folder = os.path.join(val_folder, 'labels')

    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(valid_images_folder, exist_ok=True)
    os.makedirs(valid_labels_folder, exist_ok=True)

    train_files = os.listdir(train_images_folder)

    if mode == 'video':
        video_names = set()
        for file in train_files:
            video_name = file.split('_')[1]
            video_names.add(video_name)

        num_videos = len(video_names)
        num_valid_videos = int(num_videos * (1 - ratio))
        valid_video_names = set(random.sample(video_names, num_valid_videos))

        for file in train_files:
            video_name = file.split('_')[1]

            if video_name in valid_video_names:
                shutil.move(os.path.join(train_images_folder, file), os.path.join(valid_images_folder, file))
                shutil.move(os.path.join(train_labels_folder, file.replace('.jpg', '.txt')), os.path.join(valid_labels_folder, file.replace('.jpg', '.txt')))
                print(f'move {video_name}')
    
    elif mode == 'random':
        pass


def check_and_delete_files(train_folder):
    labels_folder = os.path.join(train_folder, 'labels')
    images_folder = os.path.join(train_folder, 'images')

    img_files = os.listdir(images_folder)

    for img in img_files:
        file_name = img.split('.')[0]
        label_path = os.path.join(labels_folder, file_name + '.txt')

        if not os.path.exists(label_path):
            image_file_path = os.path.join(images_folder, img)
            os.remove(image_file_path)
            print(f"Deleted file: {image_file_path}")


def create_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)
    img_path = os.path.join(folder_path, 'images')
    label_path = os.path.join(folder_path, 'labels')
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)


def cut_two_branch_yolo(ori_folder, new_folder, cls):
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    # 遍历train和valid子文件夹
    for split_folder in ["train", "valid"]:
        ori_split_path = os.path.join(ori_folder, split_folder)
        new_split_path = os.path.join(new_folder, split_folder)

        # 遍历images和labels子文件夹
        for data_folder in ["images", "labels"]:
            new_data_path = os.path.join(new_split_path, data_folder)
            if not os.path.exists(new_data_path):
                os.makedirs(new_data_path)

        # 遍历labels文件夹中的txt文件
        label_files = glob.glob(os.path.join(ori_split_path, "labels", "*.txt"))
        for label_file in tqdm(label_files):
            
            txt_file_name = label_file.split('/')[-1]
            img_file_name = txt_file_name.split('.')[0] + '.jpg'

            # 读取txt文件内容
            with open(label_file, 'r') as f:
                lines = f.readlines()

            copy_data = []
            # 遍历每一行txt文件
            for line in lines:
                label = int(line.strip().split()[0])

                if label in cls:
                    new_label = cls.index(label)
                    new_line = re.sub(r'^\d+\b', str(new_label), line)
                    copy_data.append(new_line)

            if len(copy_data) > 0:
                new_txt_path = os.path.join(new_split_path, "labels", txt_file_name)
                with open(new_txt_path, 'w') as f:
                    for data in copy_data:
                        f.write(data)

                # 构建对应的jpg文件名
                ori_jpg_path = os.path.join(ori_split_path, "images", img_file_name)
                new_jpg_path = os.path.join(new_split_path, "images", img_file_name)

                # 复制jpg文件
                shutil.copy(ori_jpg_path, new_jpg_path)


def delete_files(folder):
    # 检查文件夹是否存在
    if os.path.exists(folder):
        # 获取文件夹中所有文件的列表
        files = os.listdir(folder)
        
        # 遍历文件列表并删除每个文件
        for file in files:
            file_path = os.path.join(folder, file)
            shutil.rmtree(file_path)

        print(f"已删除文件夹 {folder} 中的所有文件")
    else:
        print(f"文件夹 {folder} 不存在")


if __name__ == '__main__':
    # create yolo datasets
    img_folder = '/mnt/datasets/roadpp/rgb-images'
    train_folder = '/mnt/datasets/roadpp/train'
    val_folder = '/mnt/datasets/roadpp/valid'
    test_floder = '/mnt/datasets/roadpp/test'
    gt_file = '/mnt/datasets/roadpp/road_waymo_trainval_v1.0.json'

    # create_folder(train_folder)
    # create_folder(val_folder)

    # img_to_yolo(img_folder, os.path.join(train_folder, 'images'))
    # move_img_to_train_img()
    # gt_to_yolo(gt_file)
    # check_and_delete_files(train_folder)
    # cut_train_valid(train_folder, val_folder, mode='video', ratio=0.9)


    # create two branch yolo datasets
    ori_folder = '/datasets/roadpp'
    major_cls_folder = '/datasets/roadpp/two_branch_yolo/major_class'
    rare_cls_folder = '/datasets/roadpp/two_branch_yolo/rare_class'

    delete_files(major_cls_folder)
    cut_two_branch_yolo(
        ori_folder=ori_folder,
        new_folder=major_cls_folder,
        cls=[0, 1]
    )

    # delete_files(rare_cls_folder)
    # cut_two_branch_yolo(
    #     ori_folder=ori_folder,
    #     new_folder=rare_cls_folder,
    #     cls=[2, 3, 4, 5, 6, 7, 8, 9]
    # )

    
