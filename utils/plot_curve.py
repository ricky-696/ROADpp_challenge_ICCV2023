import pandas as pd
import matplotlib.pyplot as plt


def draw_loss(epochs, train, val, xlabel, ylabel):
    plt.plot(epochs, train, label='train')
    plt.plot(epochs, val, label='val')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    min_train_loss, min_train_epoch = train.min(), data.loc[train.idxmin(), 'epoch']
    min_val_loss, min_val_epoch = val.min(), data.loc[val.idxmin(), 'epoch']

    # plot min loss point
    plt.plot(min_train_epoch, min_train_loss, marker='o', markersize=10)
    plt.text(min_train_epoch, min_train_loss, f'Min: {min_train_loss:.4f}',{'fontsize':12})

    plt.plot(min_val_epoch, min_val_loss, marker='o', markersize=10)
    plt.text(min_val_epoch, min_val_loss,f'Min: {min_val_loss:.4f}',{'fontsize':12})



if __name__ == '__main__':
    csv_path = '/home/Ricky/0_Project/ROADpp_challenge_ICCV2023/runs/detect/yolov8l_T1_1920_batch_26_/results.csv'
    out_file_name = 'curve_1920_1280.jpg'
    data = pd.read_csv(csv_path).rename(columns=lambda x: x.strip())

    epochs = data['epoch']
    train_box_loss = data['train/box_loss']
    train_cls_loss = data['train/cls_loss']
    train_dfl_loss = data['train/dfl_loss']
    val_box_loss = data['val/box_loss']
    val_cls_loss = data['val/cls_loss']
    val_dfl_loss = data['val/dfl_loss']
    metrics_precision = data['metrics/precision(B)']
    metrics_recall = data['metrics/recall(B)']
    metrics_mAP50 = data['metrics/mAP50(B)']
    metrics_mAP50_95 = data['metrics/mAP50-95(B)']

    # 繪製曲線圖
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    draw_loss(data['epoch'], data['train/box_loss'], data['val/box_loss'], 'Epoch', 'Box Loss')

    plt.subplot(2, 2, 2)
    draw_loss(data['epoch'], data['train/cls_loss'], data['val/cls_loss'], 'Epoch', 'Cls Loss')

    plt.subplot(2, 2, 3)
    draw_loss(data['epoch'], data['train/dfl_loss'], data['val/dfl_loss'], 'Epoch', 'DFL Loss')


    plt.subplot(2, 2, 4)
    # plt.plot(data['epoch'], data['metrics/precision(B)'], label='metrics/precision(B)')
    # plt.plot(data['epoch'], data['metrics/recall(B)'], label='metrics/recall(B)')
    plt.plot(data['epoch'], data['metrics/mAP50(B)'], label='metrics/mAP50(B)')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.legend()

    max_mAP, max_mAP_epoch = data['metrics/mAP50(B)'].max(), data.loc[data['metrics/mAP50(B)'].idxmax(), 'epoch']

    plt.plot(max_mAP_epoch, max_mAP, marker='o', markersize=10)
    plt.text(max_mAP_epoch, max_mAP,f'Max_val_mAP: {max_mAP:.4f}',{'fontsize':12})

    plt.tight_layout()
    plt.savefig(out_file_name)
