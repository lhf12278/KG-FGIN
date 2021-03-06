
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from data_loader.preprocessing import RandomErasing
from data_loader.dataset import Samples4Market, Samples4Duke, SamplesSingle4Market, SamplesSingle4Duke,\
    TestSamples4Market, TestSamples4Duke, PersonReIDDataset
from data_loader.sampler import ClassUniformlySampler

class Loader:

    def __init__(self, config):
        transform_train = [
            transforms.Resize(config.image_size, interpolation=3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop(config.image_size)]
        if config.use_colorjitor:
            transform_train.append(transforms.ColorJitter(brightness=0.25, contrast=0.15, saturation=0.25, hue=0))
        transform_train.extend([transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        if config.use_rea:
            transform_train.append(RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]))
        self.transform_train = transforms.Compose(transform_train)

        self.transform_test = transforms.Compose([
            transforms.Resize(config.image_size, interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.datasets = ['market', 'duke', 'msmt17']
        self.source_dataset = config.source_dataset
        self.target_dataset = config.target_dataset
        self.target_camera = config.target_camera

        self.market_path = config.market_path
        self.duke_path = config.duke_path

        self.batchsize = config.batchsize

        self._load()

    def _load(self):
        source_samples = self._get_source_samples(self.source_dataset)
        self.source_loader = self._get_source_iter(source_samples, self.transform_train, self.batchsize)
        target_samples = self._get_target_samples(self.target_dataset, self.target_camera)
        self.ids = target_samples.ids
        self.target_loader = self._get_target_iter(target_samples, self.transform_train, self.batchsize)
        target_query_samples, target_gallery_samples = self._get_test_samples(self.target_dataset)
        self.target_query_loader = self._get_loader(target_query_samples, self.transform_test, 128)
        self.target_gallery_loader = self._get_loader(target_gallery_samples, self.transform_test, 128)

    def _get_source_samples(self, source_dataset):
        if source_dataset == 'market':
            samples = Samples4Market(self.market_path)
        elif source_dataset == 'duke':
            samples = Samples4Duke(self.duke_path)

        return samples

    def _get_target_samples(self, target_dataset, target_camera):
        if target_dataset == 'market':
            samples = SamplesSingle4Market(self.market_path, target_camera)
        elif target_dataset == 'duke':
            samples = SamplesSingle4Duke(self.duke_path, target_camera)

        return samples

    def _get_test_samples(self, dataset):
        if dataset == 'market':
            query_samples = TestSamples4Market(self.market_path).query_samples
            gallery_samples = TestSamples4Market(self.market_path).gallery_samples
        elif dataset == 'duke':
            query_samples = TestSamples4Duke(self.duke_path).query_samples
            gallery_samples = TestSamples4Duke(self.duke_path).gallery_samples
        return query_samples, gallery_samples

    def _get_source_iter(self, samples, transform, batchsize):
        dataset = PersonReIDDataset(samples.samples, transform=transform)
        loader = DataLoader(dataset, batch_size=batchsize, num_workers=0, drop_last=True,
                            sampler=ClassUniformlySampler(dataset, class_position=1, k=4))
        return loader

    def _get_target_iter(self, samples, transform, batchsize):
        dataset = PersonReIDDataset(samples.samples, transform=transform)
        loader = DataLoader(dataset, batch_size=batchsize, num_workers=0, drop_last=True,
                            sampler=ClassUniformlySampler(dataset, class_position=1, k=4))
        return loader

    def _get_loader(self, samples, transform, batch_size):

        dataset = PersonReIDDataset(samples, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=False, shuffle=False)
        return loader

class IterLoader:

    def __init__(self, loader):
        self.loader = loader
        self.iter = iter(self.loader)

    def next_one(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)