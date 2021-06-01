
import torch

from tools import CatMeter, time_now, ReIDEvaluator

def test(config, base, loader):

    base.set_eval()

    target_query_features_meter, target_query_pids_meter, target_query_cids_meter = CatMeter(), CatMeter(), CatMeter()
    target_gallery_features_meter, target_gallery_pids_meter, target_gallery_cids_meter = CatMeter(), CatMeter(), CatMeter()

    target_loaders = [loader.target_query_loader, loader.target_gallery_loader]

    print(time_now(), 'features start')

    with torch.no_grad():
        for target_loader_id, target_loader in enumerate(target_loaders):
            for target_data in target_loader:
                target_images, target_pids, target_cids = target_data
                target_images = target_images.to(base.device)
                target_features, target_bn_features, _, _ = base.feature_extractor(target_images)
                target_bn_features = target_bn_features.squeeze()

                if target_loader_id == 0:
                    target_query_features_meter.update(target_bn_features.data)
                    target_query_pids_meter.update(target_pids)
                    target_query_cids_meter.update(target_cids)
                elif target_loader_id == 1:
                    target_gallery_features_meter.update(target_bn_features.data)
                    target_gallery_pids_meter.update(target_pids)
                    target_gallery_cids_meter.update(target_cids)

    print(time_now(), 'features done')

    target_query_features = target_query_features_meter.get_val_numpy()
    target_gallery_features = target_gallery_features_meter.get_val_numpy()

    target_mAP, target_CMC = ReIDEvaluator(dist='cosine', mode=config.test_mode).evaluate(
        target_query_features, target_query_pids_meter.get_val_numpy(), target_query_cids_meter.get_val_numpy(),
        target_gallery_features, target_gallery_pids_meter.get_val_numpy(), target_gallery_cids_meter.get_val_numpy())

    return target_mAP, target_CMC[0: 20]

def test_with_graph(config, base, loader):

    base.set_eval()

    target_query_features_meter, target_query_pids_meter, target_query_cids_meter = CatMeter(), CatMeter(), CatMeter()
    target_gallery_features_meter, target_gallery_pids_meter, target_gallery_cids_meter = CatMeter(), CatMeter(), CatMeter()

    target_loaders = [loader.target_query_loader, loader.target_gallery_loader]

    print(time_now(), 'features start')

    with torch.no_grad():
        for target_loader_id, target_loader in enumerate(target_loaders):
            for target_data in target_loader:
                target_images, target_pids, target_cids = target_data
                target_images = target_images.to(base.device)
                target_features, _, target_local_features, _ = \
                    base.feature_extractor(target_images)
                target_graph_global_features = base.graph(target_local_features, target_features)
                target_graph_bn_global_features = base.classifier3(target_graph_global_features).squeeze()

                if target_loader_id == 0:
                    target_query_features_meter.update(target_graph_bn_global_features.data)
                    target_query_pids_meter.update(target_pids)
                    target_query_cids_meter.update(target_cids)
                elif target_loader_id == 1:
                    target_gallery_features_meter.update(target_graph_bn_global_features.data)
                    target_gallery_pids_meter.update(target_pids)
                    target_gallery_cids_meter.update(target_cids)

    print(time_now(), 'features done')

    target_query_features = target_query_features_meter.get_val_numpy()
    target_gallery_features = target_gallery_features_meter.get_val_numpy()

    target_mAP, target_CMC = ReIDEvaluator(dist='cosine', mode=config.test_mode).evaluate(
        target_query_features, target_query_pids_meter.get_val_numpy(), target_query_cids_meter.get_val_numpy(),
        target_gallery_features, target_gallery_pids_meter.get_val_numpy(), target_gallery_cids_meter.get_val_numpy())

    return target_mAP, target_CMC[0: 20]



