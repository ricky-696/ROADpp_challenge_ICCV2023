from ultralytics import YOLO

    
if __name__ == '__main__':
    # Parameter
    pretrain_weight = '/home/Ricky/ROADpp_challenge_ICCV2023/Pretrain/yolov8l.pt'
    imgsz = 1280
    batch_size = 8

    name = 'yolov8l_major_' + str(imgsz) + '_batch_' + str(batch_size) + '_VFL'

    # Training.
    model = YOLO(model=pretrain_weight)

    results = model.train(
        data = '/home/Ricky/ROADpp_challenge_ICCV2023/Track1/track1.yaml',
        imgsz = imgsz,
        # rect = True, # if input not square set this true
        mosaic = True,
        device = '0',
        epochs = 5,
        batch = batch_size,
        name = name
    )