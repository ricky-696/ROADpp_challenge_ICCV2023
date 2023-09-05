import argparse

def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='Track1', help='detect mode, only accept Track1 or Track2')
    parser.add_argument('--video_path', type=str, default='/mnt/datasets/roadpp/videos', help='video path')
    parser.add_argument('--yolo_path', type=str, default='runs/detect/yolov8l_T1_1280_batch_8_/weights/best.pt', help='yolo path')


    parser.add_argument('--two_branch', type=bool, default=False, help='used two branch YOLO')
    parser.add_argument('--major_path', type=str, default='runs/detect/yolov8l_T1_1280_batch_8_/weights/best.pt', help='major_yolo path')
    parser.add_argument('--rare_path', type=str, default='runs/detect/yolov8l_T1_1280_batch_8_/weights/best.pt', help='rare_yolo path')

    parser.add_argument('--devices', nargs='+', type=str, default='0', help='gpu number')

    parser.add_argument('--imgsz', type=tuple, default=(1280, 1280), help='yolo input size')
    parser.add_argument('--video_shape', type=tuple, default=(1280, 1920), help='original video resolution')
    parser.add_argument('--submit_shape', type=tuple, default=(600, 840), help='final submit shape')

    parser.add_argument('--pkl_name', type=str, default='T1_train_all.pkl', help='submit file name(*.pkl)')
    parser.add_argument('--save_res', type=bool, default=False, help='save submit file')

    # track2
    parser.add_argument('--action_detector_path', type=str, default='runs/action/best_weight.pt', help='action_detector_path')
    parser.add_argument('--loc_detector_path', type=str, default='runs/location/best_weight.pt', help='loc_detector_path')

    parser.add_argument('--t2_input_shape', type=tuple, default=(224, 224), help='t2_input_shape')
    parser.add_argument('--windows_size', type=int, default=4, help='sliding windows shape')
    

    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    args = arg_parse()
    print(args.mode)