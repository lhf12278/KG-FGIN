
import random
import torch.utils.data as data


class ClassUniformlySampler(data.sampler.Sampler):

    def __init__(self, data_source, class_position, k):
        self.data_source = data_source
        self.class_position = class_position
        self.k = k

        self.samples = self.data_source.samples
        self.class_dict = self._tuple2dict(self.samples)

    def __iter__(self):
        self.sample_list = self._generate_list(self.class_dict)
        return iter(self.sample_list)

    def __len__(self):
        return len(self.sample_list) * self.k

    def _tuple2dict(self, inputs):
        dict = {}
        for index, each_input in enumerate(inputs):
            class_index = each_input[self.class_position]
            if class_index not in list(dict.keys()):
                dict[class_index] = [index]
            else:
                dict[class_index].append(index)
        return dict

    def _generate_list(self, dict):
        sample_list = []
        dict_copy = dict.copy()
        keys = list(dict_copy.keys())
        random.shuffle(keys)
        for key in keys:
            value = dict_copy[key]
            if len(value) >= self.k:
                random.shuffle(value)
                sample_list.extend(value[0: self.k])
            else:
                value = value * self.k
                random.shuffle(value)
                sample_list.extend(value[0: self.k])

        return sample_list

class TargetSampler(data.sampler.Sampler):

    def __init__(self, data_target, class_position, v, k):
        self.data_target = data_target
        self.class_position = class_position
        self.v = v
        self.k = k

        self.samples = self.data_target.samples
        self.class_dict = self._tuple2dict(self.samples)

    def __iter__(self):
        self.sample_list = self._generate_list(self.class_dict)
        return iter(self.sample_list)

    def __len__(self):
        return len(self.sample_list)

    def _tuple2dict(self, inputs):
        dict = {}
        for index, each_input in enumerate(inputs):
            class_index = each_input[self.class_position]
            if class_index not in list(dict.keys()):
                dict[class_index] = [index]
            else:
                dict[class_index].append(index)
        return dict

    def _generate_list(self, dict):
        sample_list = []
        dict_copy = dict.copy()
        keys = list(dict_copy.keys())
        random.shuffle(keys)
        for key in keys:
            values = dict_copy[key]
            if len(values) >= (self.k + self.v):
                random.shuffle(values)
                sample_list.extend(values[0: self.k + self.v])
            else:
                values = values * (self.k + self.v)
                random.shuffle(values)
                sample_list.extend(values[0: self.k + self.v])
        return sample_list