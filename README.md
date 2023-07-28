# ROADpp_challenge_ICCV2023
- Data Preprocessing is located in the 'utils' folder.

## T1_YOLOv8_640*640_demo
![demo](T1_demo_epoch_20.gif)

## Training Curve

### T1_YOLOv8_1920*1280
![demo](curve_1920_1280.jpg)

### T1_YOLOv8_1280*1280_Mosaic
![demo](curve_1280_1280.jpg)

## ToDo

- [x] Convert Datasets to YOLO format()
- [x] Train YOLOv8 on Track1(train_YOLOv8.py)
- [x] ADD labels in 'utils/tube_processing.py'
- [ ] implement Tracklet Function
- [ ] Track2 TTnet
- [ ] New T1 datasets delete "majority class label" to solve label imbalence
- [ ] Two branch Yolo Pipeline
- [ ] Implement linear interpolation bbox function