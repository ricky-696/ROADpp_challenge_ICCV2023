from ultralytics import YOLO

    
if __name__ == '__main__':
    pretrain_weight = '/home/Ricky/0_Project/ROADpp_challenge_ICCV2023/runs/detect/yolov8l_rare_1920_batch_8_/weights/best.pt'
    model = YOLO(model=pretrain_weight)

    # valid
    results = model.val(
    data = '/home/Ricky/0_Project/ROADpp_challenge_ICCV2023/Track1/track1_rare.yaml',
    imgsz = 1920,
    device = '0',
    batch = 8,
    name = 'yolov8l_T1_rare_valid_'
    )

