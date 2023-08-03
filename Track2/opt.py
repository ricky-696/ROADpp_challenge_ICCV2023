import argparse

def arg_parse(func):
    parser = argparse.ArgumentParser()

    if func == "main":    
        parser.add_argument('--gpu_num', nargs='+', type=int, default=0, help='target gpu number')
        parser.add_argument('--parallelism', type=bool, default=False, help='Implements data parallelism')
        parser.add_argument('--dataset_path', '-dpath', default="/datasets/roadpp/Track2", help='path of dataset')
        parser.add_argument('--num_workers', '-nwork', type=int, default=8, help='path of dataset')

        parser.add_argument('--num_class', '-ncls',  default=22, help='path of dataset')
        parser.add_argument('--window_size', '-wsize',  default=4, help='path of dataset')
        parser.add_argument('--input_shape', '-inshape', nargs='+', default=(480, 720), help='path of dataset')

        parser.add_argument('--model', default='resnext', help='model name')
        parser.add_argument('--pretrain', default='', help='pretrain weight path')
        parser.add_argument('--epoch', type=int, default=100, help='number of epoch to train')
        parser.add_argument('--lr', type=float, default=1e-4, help='initial learning_rate')
        parser.add_argument("--batch_size", type=int, default=32, help='the batch for id')
        parser.add_argument("--seed", type=int, default=0, help='random seed')
        parser.add_argument("--debug", type=bool, default=False, help='debug switch')

    elif func == "pred":
        parser.add_argument('--gpu_num', type=int, default=0, help='target gpu number')
        parser.add_argument('--sample_rate', type=int, default=8533, help='sansor sample rate')
        parser.add_argument('--predict_timesteps_len', '--pred_len', type=int, default=5000, help='predict timestep after real data')
        parser.add_argument('--input_timesteps_len', '--input_len', type=int, default=8532, help='model\'s input size of real data timesteps len')
        parser.add_argument('--max_times', '--maxt', type=int, default=0, help='max secons for predict, 0 for all')
        parser.add_argument('--dataset_path', '--dpath', type=str, default='./dataset/fine_grinding/', help='dataset path')
        parser.add_argument('--weight_path', '--wpath', type=str, default='save_best_model_F_4.pt', help='trainned weight for predict in ./weight dir')

    opt = parser.parse_args()
    return opt