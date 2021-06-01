import os
import ast
import argparse
from data_loader import Loader
from core import Base, train_meta_learning, train_with_graph, test, test_with_graph
from tools import make_dirs, Logger, time_now, os_walk


def main(config):
    loader = Loader(config)
    base = Base(config, loader)
    make_dirs(base.output_path)
    make_dirs(base.save_logs_path)
    make_dirs(base.save_model_path)
    logger = Logger(os.path.join(base.save_logs_path, 'log.txt'))
    logger(config)

    if config.mode == 'train':
        if config.resume_train_epoch >= 0:
            base.resume_model(config.resume_train_epoch)
            start_train_epoch = config.resume_train_epoch
        else:

            start_train_epoch = 0

        if config.auto_resume_training_from_lastest_step:
            root, _, files = os_walk(base.save_model_path)
            if len(files) > 0:
                indexes = []
                for file in files:
                    indexes.append(int(file.replace('.pkl', '').split('_')[-1]))
                indexes = sorted(list(set(indexes)), reverse=False)
                base.resume_model(indexes[-1])
                start_train_epoch = indexes[-1]
                logger('Time: {}, automatically resume training from the latest step (model {})'.format(time_now(),
                                    indexes[-1]))

        for current_epoch in range(start_train_epoch, config.total_train_epoch):
            base.save_model(current_epoch)

            if current_epoch < config.use_graph:
                _, result = train_meta_learning(base, loader)
                logger('Time: {}; Epoch: {}; {}'.format(time_now(), current_epoch, result))
                if current_epoch + 1 >= 1 and (current_epoch + 1) % 40 == 0:
                    mAP, CMC = test(config, base, loader)
                    logger('Time: {}; Test on Target Dataset: {}, \nmAP: {} \n Rank: {}'.format(time_now(),
                                                                                                config.target_dataset,
                                                                                                mAP, CMC))
            else:
                _, result = train_with_graph(config, base, loader)
                logger('Time: {}; Epoch: {}; {}'.format(time_now(), current_epoch, result))
                if current_epoch + 1 >= 1 and (current_epoch + 1) % 5 == 0:
                    mAP, CMC = test_with_graph(config, base, loader)
                    logger('Time: {}; Test on Target Dataset: {}, \nmAP: {} \n Rank: {}'.format(time_now(),
                                                                                                config.target_dataset,
                                                                                                mAP, CMC))

    elif config.mode == 'test':
        base.resume_model(config.resume_test_model)
        mAP, CMC = test_with_graph(config, base, loader)
        logger('Time: {}; Test on Target Dataset: {}, \nmAP: {} \n Rank: {}'.format(time_now(),
                                                                                    config.target_dataset,
                                                                                    mAP, CMC))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str, default='cuda')
    parser.add_argument('--mode', type=str, default='train', help='train, test')
    parser.add_argument('--output_path', type=str, default='results/', help='path to save related informations')
    parser.add_argument('--source_dataset', type=str, default='market')
    parser.add_argument('--target_dataset', type=str, default='duke')
    parser.add_argument('--target_camera', type=int, default=4)
    parser.add_argument('--part_num', type=int, default=6)
    parser.add_argument('--market_path', type=str, default='G:\datasets\Market-1501-v15.09.15')
    parser.add_argument('--duke_path', type=str, default='G:/datasets/DukeMTMC/DukeMTMC-reID')
    parser.add_argument('--msmt17_path', type=str, default='G:\datasets/MSMT17V1')
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 192])
    parser.add_argument('--mis_align_ratio', type=float, default=0.05,
                        help='crop or pad ratio of our designed augmentation')
    parser.add_argument('--use_rea', default=True, help='use random erasing augmentation')
    parser.add_argument('--use_colorjitor', default=True, help='use random erasing augmentation')
    parser.add_argument('--use_graph', default=100, help='use random erasing augmentation')
    parser.add_argument('--lambda1', default=1.0, help='use random erasing augmentation')
    parser.add_argument('--batchsize', type=int, default=16, help='person count in a batch')
    parser.add_argument('--cnnbackbone', type=str, default='res50', help='res50, res50ibna')
    parser.add_argument('--source_pid_num', type=int, default=751)
    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 70],
                        help='milestones for the learning rate decay')
    parser.add_argument('--base_learning_rate', type=float, default=0.0035)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--total_train_epoch', type=int, default=110)
    parser.add_argument('--auto_resume_training_from_lastest_step', type=ast.literal_eval, default=True)
    parser.add_argument('--max_save_model_num', type=int, default=1, help='0 for max num is infinit')
    parser.add_argument('--resume_test_model', type=int, default=-1, help='')
    parser.add_argument('--resume_train_epoch', type=int, default=-1, help='')
    parser.add_argument('--test_mode', type=str, default='inter-camera', help='inter-camera, intra-camera, all')
    config = parser.parse_args()
    main(config)
