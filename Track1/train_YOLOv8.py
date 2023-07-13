from ultralytics import YOLO

    
if __name__ == '__main__':
    pretrain_weight = '/home/Ricky/0_Project/ROADpp_challenge_ICCV2023/Pretrain/yolov8l.pt'
    model = YOLO(model=pretrain_weight)

    # Training.
    results = model.train(
    data = '/home/Ricky/0_Project/ROADpp_challenge_ICCV2023/Track1/track1.yaml',
    imgsz = 1280,
    device = 0,
    epochs = 50,
    batch = 8,
    name = 'yolov8l_T1_1280_FL_batch_8_'
    )