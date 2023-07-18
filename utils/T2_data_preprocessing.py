import os
import cv2
import csv
import json
import glob
import random
import shutil


def print_bug(frame_img, annos):
    agent_id, action_id, loc_id, tube_uid = annos['agent_ids'][0], annos['action_ids'][0], annos['loc_ids'][0],annos['tube_uid'][0]

    if tube_uid == '1' or tube_uid == '2' or tube_uid == '0':
        x1, y1, x2, y2 = annos['box']
        x1_pixel, y1_pixel, x2_pixel, y2_pixel = int(x1 * img_width), int(y1 * img_height), int(x2 * img_width), int(y2 * img_height)

        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        cv2.rectangle(frame_img, (x1_pixel, y1_pixel), (x2_pixel, y2_pixel), color[int(tube_uid)], 2)

        text = f'tube_uid: {tube_uid}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        cv2.putText(frame_img, text, (x1_pixel, y1_pixel - 10), font, font_scale, color[int(tube_uid)], font_thickness)
        cv2.imwrite('bug.jpg', frame_img)



def remove_all_csv(folder_path):
    print('remove all csv...')
    csv_files = glob.glob(os.path.join(folder_path, '**/*.csv'), recursive=True)
    for file_path in csv_files:
        os.remove(file_path)
        print('removed: ', file_path)

    csv_files = glob.glob(os.path.join(folder_path, '**/*.csv'), recursive=True)
    if len(csv_files) == 0:
        print("Folder and subfolders do not contain any CSV files.")
    else:
        print("Folder and subfolders contain CSV files.")
        print("Number of CSV files found:", len(csv_files))
        input()


def remove_all_local(folder_path):

    print('remove local folder and files...')
    local_folders = glob.glob(os.path.join(folder_path, '**/local'), recursive=True)

    for folder_path in local_folders:
        shutil.rmtree(folder_path)
    
    print('create empty local folder...')
    video_folders = glob.glob(os.path.join(folder_path, '*'))
    for video_folder in video_folders:
        local_folder = os.path.join(video_folder, 'local')
        if not os.path.exists(local_folder):
            os.makedirs(local_folder)


if __name__ == '__main__':
    # create local img(cut bbox img in every frame)
    agent_labels = ['Ped', 'Car', 'Cyc', 'Mobike', 'SmalVeh', 'MedVeh', 'LarVeh', 'Bus', 'EmVeh', 'TL']
    img_folder = '/datasets/roadpp/Track2'
    gt_file = '/datasets/roadpp/road_waymo_trainval_v1.0.json'
    incomplete_labels = 0

    # remove_all_local(img_folder)
    # remove_all_csv(img_folder)

    print('Loading json file...')
    with open(gt_file, 'r') as f:
        gt_dict = json.load(f)
    

    # tube_id_dict = {}
    for video_name, video in gt_dict['db'].items():
        video_folder = os.path.join(img_folder, video_name)

        for frame_id, data in video['frames'].items():
            frame_path = os.path.join(video_folder, 'global', str(frame_id).zfill(5) + '.jpg')
            img_width, img_height = data['width'], data['height']

            if 'annos' in data:
                frame_img = cv2.imread(frame_path)

                for box_id, annos in data['annos'].items():
                    try:
                        agent_id, action_id, loc_id, tube_uid = annos['agent_ids'][0], annos['action_ids'][0], annos['loc_ids'][0], annos['tube_uid'][0]
                        # tube_id_dict[tube_uid] = True
                        local_img_path = os.path.join(video_folder, 'local', str(agent_id) + '_' + agent_labels[agent_id], tube_uid)

                        if frame_id == '4':
                            # frame_img = cv2.imread('bug.jpg')
                            print_bug(frame_img, annos)

                        if not os.path.exists(local_img_path):
                            os.makedirs(local_img_path)

                        x1, y1, x2, y2 = annos['box']
                        x1_pixel, y1_pixel, x2_pixel, y2_pixel = int(x1 * img_width), int(y1 * img_height), int(x2 * img_width), int(y2 * img_height)
                        local_img = frame_img[y1_pixel : y2_pixel, x1_pixel : x2_pixel]

                        write_img_path = os.path.join(local_img_path, str(frame_id).zfill(5) + '.jpg')
                        cv2.imwrite(write_img_path, local_img)
                        print('writeing local img: ', write_img_path)

                        # csv field: frame_id, x1, y1, x2, y2, agent_id, action_id, loc_id, tube_id
                        write_csv_path = os.path.join(local_img_path, 'label.csv')
                        with open(write_csv_path, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([str(frame_id).zfill(5), x1, y1, x2, y2, agent_id, action_id, loc_id, tube_uid])
                            print('writeing local csv: ', write_csv_path)

                    except IndexError as e:
                        incomplete_labels += 1
                        print(e)
                        print('incomplete_labels in: ', os.path.join(video_folder, 'local', str(agent_id) + '_' + agent_labels[agent_id], tube_uid))

    print('The number of incomplete_labels: ', incomplete_labels)
                    

