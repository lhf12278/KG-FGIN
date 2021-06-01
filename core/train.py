
import torch
import torch.nn as nn
from collections import OrderedDict
from data_loader import IterLoader
from tools import MultiItemAverageMeter

def train_meta_learning(base, loaders):

    base.set_train()
    source_loader = loaders.source_loader
    target_loader = loaders.target_loader
    meter = MultiItemAverageMeter()
    for i in range(400):
        source_imgs, source_pids, source_cids = IterLoader(source_loader).next_one()
        source_imgs, source_pids, source_cids = source_imgs.to(base.device), source_pids.to(base.device), \
                                                source_cids.to(base.device)
        source_features, source_bn_features, source_local_features, source_bn_local_features = base.feature_extractor(source_imgs)
        source_cls_score = base.classifier1(source_bn_features)
        source_ide_loss = base.source_ide_creiteron(source_cls_score, source_pids)
        grad = torch.autograd.grad(source_ide_loss, nn.ModuleList([base.feature_extractor,
                                                                    base.classifier1]).parameters(), retain_graph=True)
        fast_weights = list(map(lambda p: p[1] - 0.0005 * p[0], zip(grad, nn.ModuleList([base.feature_extractor,
                                                                    base.classifier1]).parameters())))
        feature_extractor_param_keys = []
        for k, v in base.feature_extractor.state_dict().items():
            if k.split('.')[-1] == 'weight' or k.split('.')[-1] == 'bias':
                feature_extractor_param_keys.append(k)
        feature_extractor_param_dict = OrderedDict()
        for i in range(len(fast_weights) - 1):
            feature_extractor_param_dict[feature_extractor_param_keys[i]] = fast_weights[i]
        base.feature_extractor.state_dict().update(feature_extractor_param_dict)

        target_imgs, target_pids, target_cids = IterLoader(target_loader).next_one()
        target_imgs, target_pids, target_cids = target_imgs.to(base.device), target_pids.to(base.device), \
                                                target_cids.to(base.device)
        target_features, target_bn_features, _, _ = base.feature_extractor(target_imgs)
        target_cls_score = base.classifier2(target_bn_features)
        target_ide_loss = base.target_ide_creiteron(target_cls_score, target_pids)
        total_loss = source_ide_loss + target_ide_loss
        base.feature_extractor_optimizer.zero_grad()
        base.classifier1_optimizer.zero_grad()
        base.classifier2_optimizer.zero_grad()
        total_loss.backward()
        base.feature_extractor_optimizer.step()
        base.classifier1_optimizer.step()
        base.classifier2_optimizer.step()
        meter.update({'source_ide_loss': source_ide_loss.data, 'target_ide_loss': target_ide_loss.data})


    return meter.get_val(), meter.get_str()

def train_multi_view(config, base, loaders):

    base.set_train()
    target_loader = loaders.target_loader
    meter = MultiItemAverageMeter()
    for i in range(100):
        target_imgs, target_pids, target_cids = IterLoader(target_loader).next_one()
        target_imgs, target_pids, target_cids = target_imgs.to(base.device), target_pids.to(base.device), \
                                                target_cids.to(base.device)
        target_features, target_bn_features, target_local_features, target_bn_local_features = \
            base.feature_extractor(target_imgs)
        target_graph_global_features = base.graph(target_local_features, target_features)
        target_cls_score = base.classifier2(target_bn_features)
        target_local_cls_score = base.local_classifier(target_bn_local_features)
        target_graph_cls_score = base.classifier3(target_graph_global_features)
        target_ide_loss = base.target_ide_creiteron(target_cls_score, target_pids)
        target_local_ide_loss = base.compute_local_pid_loss(target_local_cls_score, target_pids)
        target_graph_ide_loss = base.target_ide_creiteron(target_graph_cls_score, target_pids)

        total_loss = target_ide_loss + target_local_ide_loss + config.lambda1 * target_graph_ide_loss

        base.feature_extractor_optimizer.zero_grad()
        base.graph_optimizer.zero_grad()
        base.classifier2_optimizer.zero_grad()
        base.local_classifier_optimizer.zero_grad()
        base.classifier3_optimizer.zero_grad()
        total_loss.backward()
        base.feature_extractor_optimizer.step()
        base.graph_optimizer.step()
        base.classifier2_optimizer.step()
        base.local_classifier_optimizer.step()
        base.classifier3_optimizer.step()
        meter.update({'target_ide_loss': target_ide_loss.data,
                      'target_local_ide_loss': target_local_ide_loss.data,
                      'target_graph_ide_loss': target_graph_ide_loss.data})


    return meter.get_val(), meter.get_str()

def train_with_graph(config, base, loaders):

    base.set_train()
    target_loader = loaders.target_loader
    meter = MultiItemAverageMeter()
    for i in range(100):
        target_imgs, target_pids, target_cids = IterLoader(target_loader).next_one()
        target_imgs, target_pids, target_cids = target_imgs.to(base.device), target_pids.to(base.device), \
                                                target_cids.to(base.device)
        target_features, target_bn_features, target_local_features, target_bn_local_features = \
            base.feature_extractor(target_imgs)
        target_graph_global_features = base.graph(target_local_features, target_features)
        target_cls_score = base.classifier2(target_bn_features)
        target_local_cls_score = base.local_classifier(target_bn_local_features)
        target_graph_cls_score = base.classifier3(target_graph_global_features)
        target_ide_loss = base.target_ide_creiteron(target_cls_score, target_pids)
        target_local_ide_loss = base.compute_local_pid_loss(target_local_cls_score, target_pids)
        target_graph_ide_loss = base.target_ide_creiteron(target_graph_cls_score, target_pids)

        total_loss = target_ide_loss + target_local_ide_loss + config.lambda1 * target_graph_ide_loss

        base.feature_extractor_optimizer.zero_grad()
        base.graph_optimizer.zero_grad()
        base.classifier2_optimizer.zero_grad()
        base.local_classifier_optimizer.zero_grad()
        base.classifier3_optimizer.zero_grad()
        total_loss.backward()
        base.feature_extractor_optimizer.step()
        base.graph_optimizer.step()
        base.classifier2_optimizer.step()
        base.local_classifier_optimizer.step()
        base.classifier3_optimizer.step()
        meter.update({'target_ide_loss': target_ide_loss.data,
                      'target_local_ide_loss': target_local_ide_loss.data,
                      'target_graph_ide_loss': target_graph_ide_loss.data})


    return meter.get_val(), meter.get_str()
