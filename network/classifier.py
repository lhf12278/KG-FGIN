import torch.nn as nn

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class Classifier1(nn.Module):

    def __init__(self, in_dim, class_num):
        super(Classifier1, self).__init__()

        self.in_dim = in_dim
        self.class_num = class_num

        self.classifier = nn.Linear(self.in_dim, self.class_num, bias=False)

        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        cls_score = self.classifier(x.squeeze())
        return cls_score

class Classifier2(nn.Module):

    def __init__(self, in_dim, class_num):
        super(Classifier2, self).__init__()

        self.in_dim = in_dim
        self.class_num = class_num

        self.classifier = nn.Linear(self.in_dim, self.class_num, bias=False)

        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        cls_score = self.classifier(x.squeeze())
        return cls_score

class LocalClassifiers(nn.Module):

    def __init__(self, in_dim, part_num, pid_num):
        super(LocalClassifiers, self).__init__()

        self.in_dim = in_dim
        self.part_num = part_num
        self.pid_num = pid_num

        for i in range(self.part_num):
            setattr(self, 'classifier_{}'.format(i), Classifier2(self.in_dim, self.pid_num))

    def __call__(self, local_feature_list):
        assert len(local_feature_list) == self.part_num

        cls_score_list = []
        for i in range(self.part_num):
            local_feature_i = local_feature_list[i]
            classifier_i = getattr(self, 'classifier_{}'.format(i))
            cls_score_i = classifier_i(local_feature_i)
            cls_score_list.append(cls_score_i)

        return cls_score_list

class GraphClassifier(nn.Module):

    def __init__(self, in_dim, class_num):
        super(GraphClassifier, self).__init__()

        self.in_dim = in_dim
        self.class_num = class_num

        self.bn = nn.BatchNorm1d(self.in_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.in_dim, self.class_num, bias=False)

        self.bn.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        feature = self.bn(x.squeeze())
        cls_score = self.classifier(feature)
        if self.training:
            return cls_score
        else:
            return feature