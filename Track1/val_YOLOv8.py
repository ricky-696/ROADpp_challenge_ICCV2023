from ultralytics import YOLO

    
if __name__ == '__main__':
    pretrain_weight = '/home/Ricky/0_Project/ROADpp_challenge_ICCV2023/runs/detect/yolov8l_T1_1920_batch_26_/weights/best.pt'
    model = YOLO(model=pretrain_weight)

    # valid
    results = model.val(
    data = '/home/Ricky/0_Project/ROADpp_challenge_ICCV2023/Track1/track1.yaml',
    imgsz = [1280, 1920],
    device = '1',
    batch = 16,
    name = 'yolov8l_T1_1920_valid_'
    )

