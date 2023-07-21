import cv2
from ultralytics import YOLO


def main(model, video_path):
    tracker = model.track(source = video_path,
                          imgsz = [1280, 1280],
                          device = devices)
    print(tracker)



if __name__ == '__main__':
    video_path = '/mnt/datasets/roadpp/videos/train_00002.mp4'
    model_path = '/home/Ricky/0_Project/ROADpp_challenge_ICCV2023/runs/detect/yolov8l_T1_1280_batch_8_re/weights/best.pt'
    devices = '1'
    imgsz = [1280, 1280]
    yolo_1280 = YOLO(model_path)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter('test.mp4', 
                            fourcc, 5, (1920, 1280))
