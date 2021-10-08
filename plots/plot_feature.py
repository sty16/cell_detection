
import torch
from mmcv.parallel import collate, scatter
import matplotlib.pyplot as plt
import numpy as np
import cv2

from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmdet.datasets.pipelines import Compose

def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:, 0, :, :] * 0
    heatmaps = []
    for c in range(feature_map.shape[1]):
        heatmap += feature_map[:, c, :, :]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmaps.append(heatmap)
    return heatmaps

def draw_feature_map(features, save_dir='feature_map', name=None):
    for i, heat_maps in enumerate(features):
        heatmaps = featuremap_2_heatmap(heat_maps)
        # 这里的h,w指的是你想要把特征图resize成多大的尺寸
        # heatmap = cv2.resize(heatmap, (h, w))
        for heatmap in heatmaps:
            heatmap = np.uint8(255 * heatmap)
            # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
            #heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = heatmap
            plt.figure(0)
            im = plt.imshow(superimposed_img, cmap='rainbow')
            plt.colorbar(im)
            plt.savefig(f'feature{i}')
            plt.close(0)


if __name__ == '__main__':
    config_file = '../configs/retinanet/retinanet_r50_fpn_1x_coco.py'
    checkpoint_file = '../work_dirs/latest.pth'
    img = '../data/coco/val2017/ER0181_234.jpg'

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    result = inference_detector(model, img)
    device = next(model.parameters()).device  # model device
    cfg = model.cfg
    test_pipeline = Compose(cfg.data.test.pipeline)
    imgs = [img]
    datas = []
    for img in imgs:
        data = dict(img_info=dict(filename=img), img_prefix=None)
        data = test_pipeline(data)
        datas.append(data)
    data = collate(datas, samples_per_gpu=len(imgs))
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    # test a single image and show the results
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    x = data['img'][0]
    print(x.shape)
    feats = model.extract_feat(x)
    draw_feature_map(feats)
    # visualize the results in a new window
    show_result_pyplot(model, img, result, score_thr=0.5)