import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from bisect import bisect_right

from network import Res50BNNeck, Res50IBNaBNNeck, GraphConvNet, Classifier1, Classifier2, \
    LocalClassifiers, GraphClassifier
from tools import CrossEntropyLabelSmooth, os_walk

linked_edges = [[6, 0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5]]
adj = np.eye(7) * 1
for i, j in linked_edges:
    adj[i, j] = 1.0
    adj[j, i] = 1.0
adj = torch.from_numpy(adj.astype(np.float32))

class Base:

    def __init__(self, config, loader):
        self.config = config
        self.ids = loader.ids
        self.cnnbackbone = config.cnnbackbone
        self.source_pid_num = config.source_pid_num
        self.part_num = config.part_num
        self.max_save_model_num = config.max_save_model_num
        self.output_path = config.output_path
        self.save_model_path = os.path.join(self.output_path, 'models/')
        self.save_logs_path = os.path.join(self.output_path, 'logs/')
        self.base_learning_rate = config.base_learning_rate
        self.weight_decay = config.weight_decay
        self.milestones = config.milestones

        self._init_device()
        self._init_model()
        self._init_creiteron()
        self._init_optimizer()

    def _init_device(self):
        self.device = torch.device('cuda')

    def _init_model(self):
        if self.cnnbackbone == 'res50':
            self.feature_extractor = Res50BNNeck(self.part_num)
            self.feature_extractor = nn.DataParallel(self.feature_extractor).to(self.device)

        elif self.cnnbackbone == 'res50ibna':
            self.feature_extractor = Res50IBNaBNNeck(self.part_num)
            self.feature_extractor = nn.DataParallel(self.feature_extractor).to(self.device)
        self.graph = GraphConvNet(2048, 2048, adj)
        self.graph = nn.DataParallel(self.graph).to(self.device)
        self.classifier1 = Classifier1(2048, class_num=self.source_pid_num)
        self.classifier1 = nn.DataParallel(self.classifier1).to(self.device)
        self.classifier2 = Classifier2(2048, class_num=len(self.ids))
        self.classifier2 = nn.DataParallel(self.classifier2).to(self.device)
        self.local_classifier = LocalClassifiers(2048, self.part_num, len(self.ids))
        self.local_classifier = nn.DataParallel(self.local_classifier).to(self.device)
        self.classifier3 = GraphClassifier(2048, class_num=len(self.ids))
        self.classifier3 = nn.DataParallel(self.classifier3).to(self.device)


    def _init_creiteron(self):
        self.source_ide_creiteron = CrossEntropyLabelSmooth()
        self.target_ide_creiteron = CrossEntropyLabelSmooth()

    def compute_local_pid_loss(self, score_list, pids):
        loss_all = 0
        for i, score_i in enumerate(score_list):
            loss_i = self.target_ide_creiteron(score_i, pids)
            loss_all += loss_i
        loss_all = loss_all / float(self.part_num)
        return loss_all

    def _init_optimizer(self):
        self.feature_extractor_optimizer = optim.Adam(self.feature_extractor.parameters(),
                                                      lr=self.base_learning_rate, weight_decay=self.weight_decay)
        self.feature_extractor_lr_scheduler = WarmupMultiStepLR(self.feature_extractor_optimizer, self.milestones,
                                                                gamma=0.1, warmup_factor=0.01, warmup_iters=10)
        self.graph_optimizer = optim.Adam(self.graph.parameters(), lr=self.base_learning_rate,
                                          weight_decay=self.weight_decay)
        self.graph_lr_scheduler = WarmupMultiStepLR(self.graph_optimizer, self.milestones, gamma=0.1,
                                                    warmup_factor=0.01, warmup_iters=10)
        self.classifier1_optimizer = optim.Adam(self.classifier1.parameters(), lr=self.base_learning_rate,
                                                weight_decay=self.weight_decay)
        self.classifier1_lr_scheduler = WarmupMultiStepLR(self.classifier1_optimizer, self.milestones, gamma=0.1,
                                                          warmup_factor=0.01, warmup_iters=10)
        self.classifier2_optimizer = optim.Adam(self.classifier2.parameters(), lr=self.base_learning_rate,
                                                weight_decay=self.weight_decay)
        self.classifier2_lr_scheduler = WarmupMultiStepLR(self.classifier2_optimizer, self.milestones, gamma=0.1,
                                                          warmup_factor=0.01, warmup_iters=10)
        self.local_classifier_optimizer = optim.Adam(self.local_classifier.parameters(), lr=self.base_learning_rate,
                                                weight_decay=self.weight_decay)
        self.local_classifier_lr_scheduler = WarmupMultiStepLR(self.local_classifier_optimizer, self.milestones,
                                                               gamma=0.1, warmup_factor=0.01, warmup_iters=10)
        self.classifier3_optimizer = optim.Adam(self.classifier3.parameters(), lr=self.base_learning_rate,
                                                weight_decay=self.weight_decay)
        self.classifier3_lr_scheduler = WarmupMultiStepLR(self.classifier3_optimizer, self.milestones, gamma=0.1,
                                                          warmup_factor=0.01, warmup_iters=10)

    def save_model(self, save_epoch):
        feature_file_path = os.path.join(self.save_model_path, 'feature_{}.pkl'.format(save_epoch))
        torch.save(self.feature_extractor.state_dict(), feature_file_path)
        graph_file_path = os.path.join(self.save_model_path, 'graph_{}.pkl'.format(save_epoch))
        torch.save(self.graph.state_dict(), graph_file_path)
        classifier1_file_path = os.path.join(self.save_model_path, 'classifier1_{}.pkl'.format(save_epoch))
        torch.save(self.classifier1.state_dict(), classifier1_file_path)
        classifier2_file_path = os.path.join(self.save_model_path, 'classifier2_{}.pkl'.format(save_epoch))
        torch.save(self.classifier2.state_dict(), classifier2_file_path)
        local_classifier_file_path = os.path.join(self.save_model_path, 'localclassifier_{}.pkl'.format(save_epoch))
        torch.save(self.local_classifier.state_dict(), local_classifier_file_path)
        classifier3_file_path = os.path.join(self.save_model_path, 'classifier3_{}.pkl'.format(save_epoch))
        torch.save(self.classifier3.state_dict(), classifier3_file_path)
        if self.max_save_model_num > 0:
            root, _, files = os_walk(self.save_model_path)
            for file in files:
                if '.pkl' not in file:
                    files.remove(file)
            if len(files) > 6 * self.max_save_model_num:
                file_iters = sorted([int(file.replace('.pkl', '').split('_')[1]) for file in files], reverse=False)
                feature_file_path = os.path.join(root, 'feature_{}.pkl'.format(file_iters[0]))
                os.remove(feature_file_path)
                graph_file_path = os.path.join(root, 'graph_{}.pkl'.format(file_iters[0]))
                os.remove(graph_file_path)
                classifier1_file_path = os.path.join(root, 'classifier1_{}.pkl'.format(file_iters[0]))
                os.remove(classifier1_file_path)
                classifier2_file_path = os.path.join(root, 'classifier2_{}.pkl'.format(file_iters[0]))
                os.remove(classifier2_file_path)
                local_classifier_file_path = os.path.join(root, 'localclassifier_{}.pkl'.format(file_iters[0]))
                os.remove(local_classifier_file_path)
                classifier3_file_path = os.path.join(root, 'classifier3_{}.pkl'.format(file_iters[0]))
                os.remove(classifier3_file_path)

    def resume_last_model(self):
        root, _, files = os_walk(self.save_model_path)
        for file in files:
            if '.pkl' not in file:
                files.remove(file)
        if len(files) > 0:
            indexes = []
            for file in files:
                indexes.append(int(file.replace('.pkl', '').split('_')[-1]))
            indexes = sorted(list(set(indexes)), reverse=False)
            self.resume_model(indexes[-1])
            start_train_epoch = indexes[-1]
            return start_train_epoch
        else:
            return 0

    def resume_model(self, resume_epoch):
        feature_path = os.path.join(self.save_model_path, 'feature_{}.pkl'.format(resume_epoch))
        self.feature_extractor.load_state_dict(torch.load(feature_path), strict=False)
        print('Successfully resume feature_encoder from {}'.format(feature_path))
        graph_path = os.path.join(self.save_model_path, 'graph_{}.pkl'.format(resume_epoch))
        self.graph.load_state_dict(torch.load(graph_path), strict=False)
        print('Successfully resume graph from {}'.format(graph_path))
        classifier1_path = os.path.join(self.save_model_path, 'classifier1_{}.pkl'.format(resume_epoch))
        self.classifier1.load_state_dict(torch.load(classifier1_path), strict=False)
        print('Successfully resume classifier1 from {}'.format(classifier1_path))
        classifier2_path = os.path.join(self.save_model_path, 'classifier2_{}.pkl'.format(resume_epoch))
        self.classifier2.load_state_dict(torch.load(classifier2_path), strict=False)
        print('Successfully resume classifier2 from {}'.format(classifier2_path))
        local_classifier_path = os.path.join(self.save_model_path, 'localclassifier_{}.pkl'.format(resume_epoch))
        self.local_classifier.load_state_dict(torch.load(local_classifier_path), strict=False)
        print('Successfully resume local_classifier from {}'.format(local_classifier_path))
        classifier3_path = os.path.join(self.save_model_path, 'classifier3_{}.pkl'.format(resume_epoch))
        self.classifier3.load_state_dict(torch.load(classifier3_path), strict=False)
        print('Successfully resume classifier3 from {}'.format(classifier3_path))

    def set_train(self):
        self.feature_extractor = self.feature_extractor.train()
        self.graph = self.graph.train()
        self.classifier1 = self.classifier1.train()
        self.classifier2 = self.classifier2.train()
        self.local_classifier = self.local_classifier.train()
        self.classifier3 = self.classifier3.train()
        self.training = True

    def set_eval(self):
        self.feature_extractor = self.feature_extractor.eval()
        self.graph = self.graph.eval()
        self.classifier1 = self.classifier1.eval()
        self.classifier2 = self.classifier2.eval()
        self.local_classifier = self.local_classifier.eval()
        self.classifier3 = self.classifier3.eval()


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3, warmup_iters=500, warmup_method='linear', last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of " " increasing integers. Got {}", milestones)

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup method accepted got {}".format(warmup_method))
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / float(self.warmup_iters)
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
                base_lr
                * warmup_factor
                * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs
            ]