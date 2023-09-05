# ROADpp_challenge_ICCV2023

## File Tree
```
├── demo_pic_and_video
├── ROAD_Waymo_Baseline
├── Road-waymo-dataset
├── runs (Save model's weight)
│   ├── action
│   ├── detect
│   └── location
├── Track1
├── Track2
├── utils
```
## Full Pipeline
![demo](demo_pic_and_video/Full_pipeline.png)

## T1_YOLOv8_640*640_demo
![demo](demo_pic_and_video/T1_demo_epoch_20.gif)

## Training Curve

### T1_YOLOv8_1920*1280
![demo](demo_pic_and_video/curve_1920_1280.jpg)

### T1_YOLOv8_1280*1280_Mosaic
![demo](demo_pic_and_video/curve_1280_1280.jpg)

## ToDo

- [x] Convert Datasets to YOLO format()
- [x] Train YOLOv8 on Track1(train_YOLOv8.py)
- [x] implement Tracklet Function
- [x] Track2 Pipeline
- [x] Two branch Yolo Pipeline
- [x] Implement linear interpolation bbox function
- [ ] Add quick start guide
- [ ] Fix T2 interpolation bug
- [ ] Two branch T2