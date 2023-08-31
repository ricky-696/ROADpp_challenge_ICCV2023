from ultralytics import YOLO

    
if __name__ == '__main__':
    # Parameter
    pretrain_weight = 'yolov8l.pt'
    imgsz = 1920
    batch_size = 8

    name = 'yolov8l_major_' + str(imgsz) + '_batch_' + str(batch_size) + '_'

    # Training.
    model = YOLO(model=pretrain_weight)

    results = model.train(
        data = 'Track1/track1_major.yaml',
        imgsz = imgsz,
        # rect = True, # if input not square set this true
        mosaic = True,
        device = '0',
        epochs = 10,
        batch = batch_size,
        name = name
    )