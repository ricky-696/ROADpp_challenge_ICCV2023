import cv2
from ultralytics import YOLO


def out_of_range(x, y):
    if x < 0: x = 0
    if x > 1920: x = 1920
    if y < 0: y = 0
    if y > 1080: y = 1080
    return x, y


def get_box_img(frame, g):
    x1, y1, x2, y2 = int(g[0]), int(g[1]), int(g[2]), int(g[3])
    x1, y1 = out_of_range(x1, y1)
    x2, y2 = out_of_range(x2, y2)
    w, h = (x2 - x1), (y2 - y1)
    
    crop_img = frame[int(y1) : int(y1) + int(h), int(x1) : int(x1) + int(w)]
    orig_crop_img_wh = [w, h]
    crop_img = cv2.resize(crop_img, (256, 256), interpolation = cv2.INTER_CUBIC)

    return crop_img, orig_crop_img_wh
        

if __name__ == '__main__':
    video_path = '/mnt/datasets/roadpp/videos/train_00002.mp4'
    model_path = '/home/Ricky/0_Project/ROADpp_challenge_ICCV2023/runs/detect/yolov8l_T1_1280_batch_8_re/weights/best.pt'
    model = YOLO(model_path)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter('test.mp4', 
                            fourcc, 5, (1920, 1280))

    res = model(video_path, device='cpu')

    for r in res:
        res_plotted = r.plot()
        video.write(res_plotted)

    video.release()
    cv2.destroyAllWindows()
        



