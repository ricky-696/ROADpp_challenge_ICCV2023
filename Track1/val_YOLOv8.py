from ultralytics import YOLO

    
if __name__ == '__main__':
    pretrain_weight = '/home/Ricky/0_Project/ROADpp_challenge_ICCV2023/runs/detect/yolov8l_T1_1280_batch_8_re/weights/best.pt'
    model = YOLO(model=pretrain_weight)

    # valid
    results = model.val(
    data = '/home/Ricky/0_Project/ROADpp_challenge_ICCV2023/Track1/debug.yaml',
    imgsz = 1280,
    device = 'cpu',
    batch = 1,
    name = 'yolov8l_T1_1280_VFL_valid_'
    )

    print(results)

