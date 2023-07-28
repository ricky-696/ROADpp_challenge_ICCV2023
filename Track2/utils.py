import os
import glob

import matplotlib.pyplot as plt
from tqdm import tqdm



dataset_path = '/datasets/roadpp/Track2/'


def train_label_counter(dataset_path):
    act_len = {}
    labels = []
    all_video_path = glob.glob(os.path.join(dataset_path, 'train_*/local'))

    for video_path in all_video_path:
        agents = glob.glob(os.path.join(video_path, '*/*/label.csv'))
        labels += agents

    for label in labels:
        f = open(label, 'r')
        actions = f.readlines()

        pre_act = -1
        act_cont = 1
        for line in actions:
            l = line.split(',')
            cur_act = int(l[7])

            if pre_act != cur_act and pre_act != -1:
                if pre_act in act_len.keys():
                    act_len[pre_act].append(act_cont)
                else:
                    act_len[pre_act] = [act_cont]
                act_cont = 1

            pre_act = cur_act
            act_cont += 1

    x_bar = list(map(str, act_len.keys()))
    y_bar = [sum(i) for i in act_len.values()]
    plt.bar(x_bar, y_bar)
    for i in act_len.values():
        print(len(i), min(i), max(i))
    for a, b in zip(x_bar, y_bar):
        plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=7)
    plt.savefig('./test.png')
    plt.cla()

    



if __name__ == "__main__":
    train_label_counter(dataset_path=dataset_path)
