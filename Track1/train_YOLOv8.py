from ultralytics import YOLO

    
if __name__ == '__main__':
    # Parameter
    pretrain_weight = '/home/Ricky/0_Project/ROADpp_challenge_ICCV2023/Pretrain/yolov8l.pt'
    imgsz = [1280, 1280]
    batch_size = 32

    name = 'yolov8l_T1_' + str(imgsz[1]) + '_batch_' + str(batch_size) + '_QFLv7_'

    # Training.
    model = YOLO(model=pretrain_weight)

    results = model.train(
    data = '/home/Ricky/0_Project/ROADpp_challenge_ICCV2023/Track1/track1.yaml',
    imgsz = imgsz,
    # rect = True, # if input not square set this true
    device = '2,3',
    epochs = 50,
    batch = batch_size,
    name = name
    )