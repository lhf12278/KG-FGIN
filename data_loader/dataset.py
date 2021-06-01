
import os
import copy
import numpy as np
from PIL import Image

from tools import os_walk

class PersonReIDSamples:
    def __init__(self, dataset_path):

        self.dataset_path = os.path.join(dataset_path, 'bounding_box_train/')
        samples = self._load_samples(self.dataset_path)
        pids, cids, samples = self._reorder_labels(samples, 1, 2)

        self.samples = samples
        self.pids = pids
        self.cids = cids

    def _reorder_labels(self, samples, pid_index, cid_index):

        pids = []
        cids = []
        for sample in samples:
            pids.append(sample[pid_index])
            cids.append(sample[cid_index])

        pids = list(set(pids))
        pids.sort()
        cids = list(set(cids))
        cids.sort()
        for sample in samples:
            sample[pid_index] = pids.index(sample[pid_index])
            sample[cid_index] = cids.index(sample[cid_index])
        return pids, cids, samples

    def _load_samples(self, floder_dir):
        samples = []
        root_path, _, files_name = os_walk(floder_dir)
        for file_name in files_name:
            if 'jpg' in file_name:
                identity, camera = self._analysis_file_name(file_name)
                samples.append([root_path + file_name, identity, camera])

        return samples

    def _analysis_file_name(self, file_name):
        split_list = file_name.replace('.jpg', '').replace('c', '').replace('s', '_').split('_')
        identity_id, camera_id = int(split_list[0]), int(split_list[1])
        return identity_id, camera_id


class SingleCameraPersonReIDSamples:
    def __init__(self, dataset_path, camera_id):

        self.dataset_path = os.path.join(dataset_path, 'bounding_box_train/')
        samples = self._load_samples(self.dataset_path, camera_id)
        ids, samples = self._reorder_labels(samples, 1)

        self.samples = samples
        self.ids = ids

    def _reorder_labels(self, samples, label_index):

        ids = []
        for sample in samples:
            ids.append(sample[label_index])

        ids = list(set(ids))
        ids.sort()
        for sample in samples:
            sample[label_index] = ids.index(sample[label_index])
        return ids, samples

    def _load_samples(self, floder_dir, camera_id):
        label_samples = []
        root_path, _, files_name = os_walk(floder_dir)
        for file_name in files_name:
            if 'jpg' in file_name:
                identity, camera = self._analysis_file_name(file_name)
                if camera == camera_id:
                    label_samples.append([root_path + file_name, identity, camera_id])

        return label_samples

    def _analysis_file_name(self, file_name):
        split_list = file_name.replace('.jpg', '').replace('c', '').replace('s', '_').split('_')
        identity_id, camera_id = int(split_list[0]), int(split_list[1])
        return identity_id, camera_id


class TestPersonReIDSamples:
    def __init__(self, dataset_path):

        self.query_path = os.path.join(dataset_path, 'query/')
        self.gallery_path = os.path.join(dataset_path, 'bounding_box_test/')
        query_samples = self._load_samples(self.query_path)
        gallery_samples = self._load_samples(self.gallery_path)
        self.query_samples = query_samples
        self.gallery_samples = gallery_samples

    def _load_samples(self, floder_dir):
        samples = []
        root_path, _, files_name = os_walk(floder_dir)
        for file_name in files_name:
            if 'jpg' in file_name:
                identity_id, camera_id = self._analysis_file_name(file_name)
                samples.append([root_path + file_name, identity_id, camera_id])
        return samples

    def _analysis_file_name(self, file_name):
        split_list = file_name.replace('.jpg', '').replace('c', '').replace('s', '_').split('_')
        identity_id, camera_id = int(split_list[0]), int(split_list[1])
        return identity_id, camera_id


class Samples4Market(PersonReIDSamples):
    pass


class Samples4Duke(PersonReIDSamples):
    def _analysis_file_name(self, file_name):
        split_list = file_name.replace('.jpg', '').replace('c', '').split('_')
        identi_id, camera_id = int(split_list[0]), int(split_list[1])
        return identi_id, camera_id

class SamplesSingle4Market(SingleCameraPersonReIDSamples):
    pass


class SamplesSingle4Duke(SingleCameraPersonReIDSamples):
    def _analysis_file_name(self, file_name):
        split_list = file_name.replace('.jpg', '').replace('c', '').split('_')
        identi_id, camera_id = int(split_list[0]), int(split_list[1])
        return identi_id, camera_id

class TestSamples4Market(TestPersonReIDSamples):
    pass


class TestSamples4Duke(TestPersonReIDSamples):
    def _analysis_file_name(self, file_name):
        split_list = file_name.replace('.jpg', '').replace('c', '').split('_')
        identi_id, camera_id = int(split_list[0]), int(split_list[1])
        return identi_id, camera_id


class PersonReIDDataset:

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        this_sample = copy.deepcopy(self.samples[index])
        this_sample[0] = self._loader(this_sample[0])
        if self.transform is not None:
            this_sample[0] = self.transform(this_sample[0])
        this_sample[1] = np.array(this_sample[1])
        this_sample[2] = np.array(this_sample[2])
        return this_sample

    def __len__(self):
        return len(self.samples)

    def _loader(self, img_path):
        return Image.open(img_path).convert('RGB')
