import os
import copy
import time
import shutil
import random
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torch import optim
# from torch.argsim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from models.resnet import ResNet50, ResNet101, ResNet152
from models.resnext import ResNeXt29_2x64d, ResNeXt50_32x4d, ResNeXt101_32x4d, ResNeXt101_64x4d
from dataset import track2_dataset
from train_action import train, test
from opt import arg_parse

# device_ids = [0, 1]

args = arg_parse("main")
start_time = int(time.time())
train_id = "debug" if args.debug else start_time % 100000

agent_labels = ['Ped', 'Car', 'Cyc', 'Mobike', 'SmalVeh', 'MedVeh', 'LarVeh', 'Bus', 'EmVeh', 'TL']
action_labels = ['Red', 'Amber', 'Green', 'MovAway', 'MovTow', 'Mov', 'Rev', 'Brake', 'Stop', 'IncatLft', 'IncatRht', 'HazLit', 'TurLft', 'TurRht', 'MovRht', 'MovLft', 'Ovtak', 'Wait2X', 'XingFmLft', 'XingFmRht', 'Xing', 'PushObj']
loc_labels = ['VehLane', 'OutgoLane', 'OutgoCycLane', 'OutgoBusLane', 'IncomLane', 'IncomCycLane', 'IncomBusLane', 'Pav', 'LftPav', 'RhtPav', 'Jun', 'xing', 'BusStop', 'parking', 'LftParking', 'rightParking']

action_order = [3, 7, 8, 4, 13, 5, 10, 9, 17, 12, 20, 14, 18, 15, 19, 6, 11]
loc_order = [1, 0, 10, 8, 4, 14, 7, 9, 13, 15, 11, 12, 5, 6]

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


def plot_loss_img(epoch, training_loss, testing_loss, best_loss):
    epochs = [str(i) for i in range(1, epoch+1)]
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, training_loss, label="training loss")
    plt.plot(epochs, testing_loss, label="testing loss")
    plt.text(best_loss[0], best_loss[1]+0.005, 'epoch {}: {:.6f}'.format(best_loss[0]+1, best_loss[1]), fontsize=11, horizontalalignment='right', color='black')
    plt.plot(best_loss[0], best_loss[1], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")

    plt.xticks(rotation=60)
    axis = plt.gca()
    axis.xaxis.set_major_locator(ticker.MultipleLocator(base=round(epoch/10 + 1)))
    plt.title("Loss History")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./log/{}/loss_history.jpg".format(train_id))
    plt.close()


def get_mini_datasets(dataset, dataset_size):
    mini_size = [dataset_size for _ in range(int(1 / dataset_size))]
    if sum(mini_size) < 1:
        mini_size.append(1 - sum(mini_size))
    all_minibatch = data.random_split(dataset, mini_size)

    return all_minibatch


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
    if args.model == 'resnext':
        model = ResNeXt101_32x4d(num_class=args.num_class,
            input_dim=args.window_size * 3 * 2, input_size=args.input_shape)
        # model = ResNeXt101_64x4d(num_class=args.num_class,
        #     input_dim=args.window_size * 3 * 2, input_size=args.input_shape)
    if args.model == 'resnet':
        model = ResNet152(num_classes=args.num_class, channels=args.window_size * 3 * 2)
    
    if args.parallelism:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_num)
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
            num_workers = args.num_workers
        ))
    test_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = args.num_workers
    )
    
    logger.info("optimizer: Adam")
    logger.info("Loss function: CrossEntropyLoss")
    # optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    logger.info("----------------")
    logger.info("train start...")
    # first, best_loss = True, [0, 999.]  # draw or not, check best model
    best = 999.
    # epoch_iter = tqdm(range(1, args.epoch+1), initial=1, desc="Train loss: NaN| Test loss: NaN| Epoch")
    for epoch in range(1, args.epoch+1, 3):
        for idx, train_loader in enumerate(train_loaders):
            # train
            train_loss, train_acc = train(args, model, train_loader, optimizer, criterion, epoch + idx)
            # training_loss.append(train_loss)

            # testing
            test_loss, test_acc, uncorrect_count = test(args, model, epoch + idx, test_loader, criterion)
            # testing_loss.append(test_loss)

            writer.add_scalars(main_tag="Loss History", tag_scalar_dict={
                "Train_Loss": train_loss,
                "Valid_Loss": test_loss
            }, global_step=epoch + idx)
            writer.add_scalars(main_tag="Accuracy History", tag_scalar_dict={
                "Train_Acc": train_acc,
                "Valid_Acc": test_acc
            }, global_step=epoch + idx)

            fig = plt.figure()
            plt.bar(list(map(str(action_order))), uncorrect_count)
            writer.add_figure('uncorrect_count', fig)

            logger.disabled = True
            logger.info("Epoch {}, train_loss: {:.6f}, test_loss: {:.6f}".format(epoch, train_loss, test_loss))
            logger.disabled = False

            if best > test_loss:
                best = test_loss
                torch.save(model, "./runs/{}/weight/best_weight.pt".format(train_id))
                logger.info("Update best weight at epoch {}, best test loss: {:.6f}".format(epoch, test_loss))

    total_time = (int(time.time()) - start_time) // 60
    h_time, m_time = (total_time // 60), total_time % 60
    logger.info("Total prossesing time = {}:{}:{}".format(h_time, m_time, int(time.time()) - total_time*60))
    logger.info("{} process end.".format(train_id))


if __name__ == '__main__':
    main()