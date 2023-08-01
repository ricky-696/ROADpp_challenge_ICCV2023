import os
import copy
import time
import shutil
import random
import logging
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch import optim
# from torch.argsim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from models.resnet import ResNet50, ResNet101, ResNet152
from models.resnext import ResNeXt29_2x64d, ResNeXt50_32x4d, ResNeXt101_32x4d, ResNeXt101_64x4d
from models.hug import ViT
from models.hug import Swin
from dataset import track2_dataset
from train_action import train, test
from opt import arg_parse

args = arg_parse("main")
start_time = int(time.time())
train_id = "debug" if args.debug else start_time % 100000

if len(args.gpu_num) == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num[0]
else:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    gpu_string = args.gpu_num[0]
    for i in args.gpu_num[1:]:
        gpu_string = gpu_string + ", " + i
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_string
device_id = [int(i) for i in range(len(args.gpu_num))]

agent_labels = ['Ped', 'Car', 'Cyc', 'Mobike', 'SmalVeh', 'MedVeh', 'LarVeh', 'Bus', 'EmVeh', 'TL']
action_labels = ['Red', 'Amber', 'Green', 'MovAway', 'MovTow', 'Mov', 'Rev', 'Brake', 'Stop', 'IncatLft', 'IncatRht', 'HazLit', 'TurLft', 'TurRht', 'MovRht', 'MovLft', 'Ovtak', 'Wait2X', 'XingFmLft', 'XingFmRht', 'Xing', 'PushObj']
loc_labels = ['VehLane', 'OutgoLane', 'OutgoCycLane', 'OutgoBusLane', 'IncomLane', 'IncomCycLane', 'IncomBusLane', 'Pav', 'LftPav', 'RhtPav', 'Jun', 'xing', 'BusStop', 'parking', 'LftParking', 'rightParking']


def torch_init(args):
    # including random_split
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True


def logger_init(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    handler1 = logging.FileHandler("./runs/{}/train_log.log".format(train_id))
    handler2 = logging.StreamHandler()
    formatter = logging.Formatter(str(train_id) + ': %(asctime)s - %(levelname)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)

    logger.info("{} process start.".format(train_id))
    logger.info("--- option argument setting ---")
    logger.info("gpu_num = {}".format(args.gpu_num))
    logger.info("parallelism = {}".format(args.parallelism))
    logger.info("dataset_path = {}".format(args.dataset_path))
    logger.info("epoch = {}".format(args.epoch))
    logger.info("lr = {}".format(args.lr))
    logger.info("batch_size = {}".format(args.batch_size))
    logger.info("seed = {}".format(args.seed))
    logger.info("-------------------------------")

    return logger


"""
def GPU_init(args, logger):
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    devices = [i for i in range(device_count)]

    logger.info("--- GPU environment ---")
    for i in devices:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        logger.info("GPU " + str(i) + ": " + str(pynvml.nvmlDeviceGetName(handle))[2:-1])
    logger.info("-----------------------")

    visible_string = '0'
    for i in devices[1:]:
        visible_string = visible_string + ', ' + str(i)
    
    torch.cuda.set_device(args.gpu_num)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_string

    return devices
"""


def get_confusion_matrix(preds, labels, num_classes, normalize="true"):
    if isinstance(preds, list):
        preds = torch.cat(preds, dim=0)
    if isinstance(labels, list):
        labels = torch.cat(labels, dim=0)

    if labels.ndim == preds.ndim:
        labels = torch.argmax(labels, dim=-1)
    
    preds = torch.flatten(torch.argmax(preds, dim=-1))
    labels = torch.flatten(labels)

    cmtx = confusion_matrix(
        labels, preds, labels=list(range(num_classes)), normalize=normalize
    )

    return cmtx


def plot_confusion_matrix(cmtx, num_classes, cls_names=None, figsize=None):
    if cls_names is None or type(cls_names) != list:
        cls_names = [str(i) for i in range(num_classes)]

    fig = plt.figure(figsize=figsize)

    plt.imshow(cmtx, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(cls_names))
    plt.xticks(tick_marks, cls_names, rotation=45)
    plt.yticks(tick_marks, cls_names)

    threshold = cmtx.max() / 2.0
    for i, j in itertools.product(range(cmtx.shape[0]), range(cmtx.shape[1])):
        color = "white" if cmtx[i, j] > threshold else "black"
        plt.text(
            j, i, format(cmtx[i, j], ".2f") if cmtx[i, j] != 0 else ".", 
            horizontalalignment="center", color=color)
    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    return fig


def add_confusion_matrix(writer, cmtx, num_classes, global_step=None, subset_ids=None,
                         class_names=None, tag="Confusion Matrix", figsize=None):
    if subset_ids is None or len(subset_ids) != 0:
        if class_names is None:
            class_names = [str(i) for i in range(num_classes)]
        
        if subset_ids is None:
            subset_ids = list(range(num_classes))

        sub_cmtx = cmtx[subset_ids, :][:, subset_ids]
        sub_names = [class_names[j] for j in subset_ids]

        sub_cmtx = plot_confusion_matrix(
            sub_cmtx,
            num_classes=len(subset_ids),
            cls_names=sub_names,
            figsize=figsize
        )

        writer.add_figure(tag=tag, figure=sub_cmtx, global_step=global_step)


def main():
    output_path = './runs/{}/'.format(train_id)

    # check folder
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        os.mkdir(os.path.join(output_path, 'weight'))

    # init
    logger = logger_init(args)
    # if not args.cpu:
    #     gpu_list = GPU_init(args, logger)
    torch_init(args)
    writer = SummaryWriter(log_dir=output_path)

    # model
    args.input_shape = [int(args.input_shape[0]), int(args.input_shape[1])]
    if "swin" in args.model or "vit" in args.model:
        assert args.input_shape[0] == args.input_shape[1]

    if args.model == 'resnext': # broken
        model = ResNeXt101_32x4d(num_class=args.num_class,
            input_dim=args.window_size * 3 * 2, input_size=args.input_shape)
    if args.model == 'resnet':
        model = ResNet152(num_classes=args.num_class, channels=args.window_size * 3 * 2)
    if args.model == "vit":
        model = ViT.ViT_h(num_class=args.num_class, num_channels=args.window_size * 3 * 2, input_size=args.input_shape[0])
    if args.model == "swin":
        if args.input_shape[0] == 224:
            model = Swin.Swin_l_224(num_class=args.num_class, num_channels=args.window_size * 3 * 2, input_size=args.input_shape[0])
        if args.input_shape[0] == 384:
            model = Swin.Swin_l_384(num_class=args.num_class, num_channels=args.window_size * 3 * 2, input_size=args.input_shape[0])
    if args.model == "swin_t":
        model = Swin.Swin_t(num_class=args.num_class, num_channels=args.window_size * 3 * 2, input_size=args.input_shape[0])
    if args.model == "swin_s":
        model = Swin.Swin_s(num_class=args.num_class, num_channels=args.window_size * 3 * 2, input_size=args.input_shape[0])
    
    if args.parallelism:
        model = torch.nn.DataParallel(model, device_ids=device_id)
    model = model.cuda()

    logger.info("loading train data...")
    logger.info("train test split ratio: 0.7")
    logger.info("mini batch size: 0.1")

    # dataset
    full_dataset = track2_dataset(args)
    subsets = torch.utils.data.random_split(full_dataset, [0.3, 0.3, 0.3, 0.1])
    train_sets, valid_set = subsets[:-1], subsets[-1]

    train_loaders = []
    for train_set in train_sets:
        train_loaders.append(torch.utils.data.DataLoader(
            train_set, 
            batch_size = args.batch_size, 
            shuffle = True,
            num_workers = args.num_workers,
            pin_memory = False
        ))
    test_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = args.num_workers,
        pin_memory = False
    )
    
    logger.info("optimizer: Adam")
    logger.info("Loss function: CrossEntropyLoss")
    # optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    logger.info("----------------")
    logger.info("train start...")
    best = 999.
    for epoch in range(1, args.epoch+1, 3):
        for idx, train_loader in enumerate(train_loaders):
            # train
            train_loss, train_acc = train(args, model, train_loader, optimizer, criterion, epoch + idx, writer)

            # testing
            test_loss, test_acc, preds, labels = test(args, model, test_loader, criterion, epoch + idx)

            # tensorboard loss and acc
            writer.add_scalars(main_tag="Loss History", tag_scalar_dict={
                "Train_Loss": train_loss,
                "Valid_Loss": test_loss
            }, global_step=epoch + idx)
            writer.add_scalars(main_tag="Accuracy History", tag_scalar_dict={
                "Train_Acc": train_acc,
                "Valid_Acc": test_acc
            }, global_step=epoch + idx)

            # tensorboard confusion matrix
            preds = torch.cat(preds, dim=0)
            labels = torch.cat(labels, dim=0)
            cmtx = get_confusion_matrix(preds, labels, len(action_labels))
            add_confusion_matrix(writer, cmtx, num_classes=len(action_labels), 
                                 class_names=action_labels, tag="Test Confusion Matrix", figsize=[10, 8])

            logger.disabled = True
            logger.info("Epoch {}, train_loss: {:.6f}, test_loss: {:.6f}".format(epoch+idx, train_loss, test_loss))
            logger.disabled = False

            if best > test_loss:
                best = test_loss
                torch.save(model, "./runs/{}/weight/best_weight.pt".format(train_id))
                logger.info("Update best weight at epoch {}, best test loss: {:.6f}, test acc: {:.6f}".format(epoch+idx, test_loss, test_acc))
    
    writer.close()
    total_time = (int(time.time()) - start_time) // 60
    h_time, m_time = (total_time // 60), total_time % 60
    logger.info("Total prossesing time = {}:{}:{}".format(h_time, m_time, int(time.time()) - total_time*60))
    logger.info("{} process end.".format(train_id))


if __name__ == '__main__':
    main()