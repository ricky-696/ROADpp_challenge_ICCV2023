import cv2
from ultralytics import YOLO

if __name__ == '__main__':
    video_path = '/mnt/datasets/roadpp/videos/train_00000.mp4'
    model_path = '/home/Ricky/0_Project/ROADpp_challenge_ICCV2023/runs/detect/yolov8l_T1_/weights/best.pt'
    model = YOLO(model_path)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter('test.mp4', 
                            fourcc, 5, (1920, 1280))

    res = model(video_path, device=1)

    for r in res:
        res_plotted = r.plot()
        video.write(res_plotted)

    video.release()
    cv2.destroyAllWindows()
        



