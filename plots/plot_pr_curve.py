import os
import mmcv
import numpy as np
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from mmcv import Config
from mmdet.datasets import build_dataset


class PrPlot:
    def __init__(self, model, model_name, config_file, result_file, metric='bbox'):
        cfg = Config.fromfile(config_file)
        if isinstance(cfg.data.test, dict):
            cfg.data.test.test_mode = True
        elif isinstance(cfg.data.test, list):
            for ds_cfg in cfg.data.test:
                ds_cfg.test_mode = True
        dataset = build_dataset(cfg.data.test)
        # load result file in pkl format
        pkl_results = mmcv.load(result_file)
        # convert pkl file (list[list | tuple | ndarray]) to json
        json_results, _ = dataset.format_results(pkl_results)
        # initialize COCO instance
        coco = COCO(annotation_file=cfg.data.test.ann_file)
        coco_gt = coco
        coco_dt = coco_gt.loadRes(json_results[metric]) 
        # initialize COCOeval instance
        coco_eval = COCOeval(coco_gt, coco_dt, metric)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        # extract eval data
        self.precisions = coco_eval.eval["precision"]


    def plot_iou_pr_curve(self):
        '''
        precisions[T, R, K, A, M]
        T: iou thresholds [0.5 : 0.05 : 0.95], idx from 0 to 9
        R: recall thresholds [0 : 0.01 : 1], idx from 0 to 100
        K: category, idx from 0 to ...
        A: area range, (all, small, medium, large), idx from 0 to 3
        M: max dets, (1, 10, 100), idx from 0 to 2
        '''
        pr_arrays = []
        labels = ["iou=0.5", "iou=0.55", "iou=0.6", "iou=0.65", "iou=0.7", "iou=0.75", "iou=0.8", "iou=0.85", "iou=0.9", "iou=0.95"]
        x = np.arange(0.0, 1.01, 0.01)
        for i in range(10):
            pr_arrays.append(self.precisions[i, :, 0, 0, 1])
            plt.plot(x, pr_arrays[i], label=labels[i])

        plt.xlabel("recall")
        plt.ylabel("precison")
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.01)
        plt.grid(True)
        plt.legend(loc='lower left', framealpha=0.2)
        plt.savefig("plots/result_iou_ap.png")



if __name__ == "__main__":
    model = "retinanet"
    model_name = "retinanet_r50_fpn_1x_coco"

    config_file = f"configs/{model}/{model_name}.py"
    result_file = "test_result/latest.pkl"
    metric = 'bbox'

    pr_plot = PrPlot(model, model_name, config_file, result_file, metric)
    pr_plot.plot_iou_pr_curve()

    


    

