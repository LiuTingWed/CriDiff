import torch
# import SimpleITK as sitk
from scipy.spatial.distance import directed_hausdorff
from medpy import metric
import numpy as np
from numpy import *
from PIL import Image

"""
The evaluation implementation refers to the following paper:
"Selective Feature Aggregation Network with Area-Boundary Constraints for Polyp Segmentation"
https://github.com/Yuqi-cuhk/Polyp-Seg
"""
def evaluate(pred, gt):
    Recall, Specificity, Precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    MSD = []
    ASD = []
    # h = []
    len = pred.size(dim=0)

    for i in range(len):
        predict = pred[i, :].argmax(0)
        pred_binary = (predict >= 0.5).float()
        pred_binary_inverse = (pred_binary == 0).float()

        gt_binary = (gt[i] >= 0.5).float()
        gt_binary_inverse = (gt_binary == 0).float()

        a = pred_binary.unsqueeze(0).cpu()
        b = gt_binary.unsqueeze(0).cpu()
        # a = pred_binary.cpu()
        # b = gt_binary.cpu()
        MSD.append(max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0]))

        # if a.max() > 0 and b.max() > 0:
        #     ASD.append(metric.binary.asd(np.array(a), np.array(b)))
        #     h.append(metric.binary.hd95(np.array(a), np.array(b)))
        # dict = computeQualityMeasures(a,b)
        # HD.append(max(directed_hausdorff(a,b)[0],directed_hausdorff(b,a)[0]))
        # HD.append(dict["Hausdorff"])

        TP = pred_binary.mul(gt_binary).sum()
        FP = pred_binary.mul(gt_binary_inverse).sum()
        TN = pred_binary_inverse.mul(gt_binary_inverse).sum()
        FN = pred_binary_inverse.mul(gt_binary).sum()

        if TP.item() == 0:
            # print('TP=0 now!')
            # print('Epoch: {}'.format(epoch))
            # print('i_batch: {}'.format(i_batch))
            TP = torch.Tensor([1]).cuda()

        # recall
        recall = TP / (TP + FN)

        # Specificity or true negative rate
        specificity = TN / (TN + FP)

        # Precision or positive predictive value
        precision = TP / (TP + FP)

        # F1 score = Dice
        f1 = 2 * precision * recall / (precision + recall)

        # F2 score
        f2 = 5 * precision * recall / (4 * precision + recall)

        # Overall accuracy
        aCC_overall = (TP + TN) / (TP + FP + FN + TN)

        # IoU for poly
        ioU_poly = TP / (TP + FP + FN)

        # IoU for background
        ioU_bg = TN / (TN + FP + FN)

        # mean IoU
        ioU_mean = (ioU_poly + ioU_bg) / 2.0

        Recall = Recall + recall
        Specificity = Specificity + specificity
        Precision = Precision + precision
        F1 = F1 + f1
        F2 = F2 + f2
        ACC_overall = ACC_overall + aCC_overall
        IoU_poly = IoU_poly + ioU_poly
        IoU_bg = IoU_bg + ioU_bg
        IoU_mean = IoU_mean + ioU_mean

    return Recall / len, Specificity / len, Precision / len, F1 / len, F2 / len, ACC_overall / len, IoU_poly / len, IoU_bg / len, IoU_mean / len, mean(
        MSD), mean(ASD)

def evaluate_prostate(pred, gt):


    quality = dict()
    pred = torch.max(pred, dim = 1).indices

    Pred_bg = (pred == 0).float()
    Pred_pz = (pred == 1).float()
    Pred_cg = (pred == 2).float()
    Gt_bg = (gt == 0).float()
    Gt_pz = (gt == 1).float()
    Gt_cg = (gt == 2).float()

    Pred_bg_inverse = (Pred_bg == 0).float()
    Pred_pz_inverse = (Pred_pz == 0).float()
    Pred_cg_inverse = (Pred_cg == 0).float()
    Gt_bg_inverse = (Gt_bg == 0).float()
    Gt_pz_inverse = (Gt_pz == 0).float()
    Gt_cg_inverse = (Gt_cg == 0).float()

    TP_BG = Pred_bg.mul(Gt_bg).sum()
    FP_BG = Pred_bg.mul(Gt_bg_inverse).sum()
    TN_BG = Pred_bg_inverse.mul(Gt_bg_inverse).sum()
    FN_BG = Pred_bg_inverse.mul(Gt_bg).sum()
    if TP_BG.item() == 0:
        # print('TP=0 now!')
        # print('Epoch: {}'.format(epoch))
        # print('i_batch: {}'.format(i_batch))
        TP_BG = torch.Tensor([1]).cuda()
    Recall_BG= TP_BG / (TP_BG + FN_BG)
    Precision_BG = TP_BG / (TP_BG + FP_BG)
    quality["BG_Dice"] = 2 * Precision_BG * Recall_BG / (Precision_BG + Recall_BG)

    TP_PZ = Pred_pz.mul(Gt_pz).sum()
    FP_PZ = Pred_pz.mul(Gt_pz_inverse).sum()
    TN_PZ = Pred_pz_inverse.mul(Gt_pz_inverse).sum()
    FN_PZ = Pred_pz_inverse.mul(Gt_pz).sum()
    if TP_PZ.item() == 0:
        # print('TP=0 now!')
        # print('Epoch: {}'.format(epoch))
        # print('i_batch: {}'.format(i_batch))
        TP_PZ = torch.Tensor([1]).cuda()
    Recall_PZ = TP_PZ / (TP_PZ + FN_PZ)
    Precision_PZ = TP_PZ / (TP_PZ + FP_PZ)
    quality["PZ_Dice"] = 2 * Precision_PZ * Recall_PZ / (Precision_PZ + Recall_PZ)

    TP_CG = Pred_cg.mul(Gt_cg).sum()
    FP_CG = Pred_cg.mul(Gt_cg_inverse).sum()
    TN_CG = Pred_cg_inverse.mul(Gt_cg_inverse).sum()
    FN_CG = Pred_cg_inverse.mul(Gt_cg).sum()
    if TP_CG.item() == 0:
        # print('TP=0 now!')
        # print('Epoch: {}'.format(epoch))
        # print('i_batch: {}'.format(i_batch))
        TP_CG = torch.Tensor([1]).cuda()
    Recall_CG = TP_CG / (TP_CG + FN_CG)
    Precision_CG = TP_CG / (TP_CG + FP_CG)
    quality["CG_Dice"] = 2 * Precision_CG * Recall_CG / (Precision_CG + Recall_CG)


    # quality["PZ_avgHausdorff"] = quality["PZ_avgHausdorff"] / Pred_pz.shape[0]
    # quality["PZ_Hausdorff"] = quality["PZ_Hausdorff"] / Pred_pz.shape[0]
    # quality["PZ_dice"] = quality["PZ_dice"] / Pred_pz.shape[0]

    # quality["CG_avgHausdorff"] = quality["CG_avgHausdorff"] / Pred_pz.shape[0]
    # quality["CG_Hausdorff"] = quality["CG_Hausdorff"] / Pred_pz.shape[0]
    # quality["CG_dice"] = quality["CG_dice"] / Pred_pz.shape[0]
    return quality

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def evaluate_prostateSingle(pred, gt):
    num_classes = 1+1
    output_seg = pred.argmax(1)
    target = gt
    axes = tuple(range(1, len(target.shape)))
    tp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
    fp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
    fn_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
    for c in range(1, num_classes):
        tp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target == c).float(), axes=axes)
        fp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target != c).float(), axes=axes)
        fn_hard[:, c - 1] = sum_tensor((output_seg != c).float() * (target == c).float(), axes=axes)

    tp_hard = tp_hard.sum(0, keepdim=False).detach().cpu().numpy()
    fp_hard = fp_hard.sum(0, keepdim=False).detach().cpu().numpy()
    fn_hard = fn_hard.sum(0, keepdim=False).detach().cpu().numpy()

    online_eval_foreground_dc = list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8))
    online_eval_tp = list(tp_hard)
    online_eval_fp = list(fp_hard)
    online_eval_fn = list(fn_hard)

    return online_eval_foreground_dc, online_eval_tp, online_eval_fp, online_eval_fn


def computeQualityMeasures(lP, lT):
    quality = dict()
    labelPred = sitk.GetImageFromArray(lP, isVector=False)
    labelTrue = sitk.GetImageFromArray(lT, isVector=False)
    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    # hausdorffcomputer.Execute(labelTrue > 0.5, labelPred > 0.5)
    quality["avgHausdorff"] = hausdorffcomputer.GetAverageHausdorffDistance()
    quality["Hausdorff"] = hausdorffcomputer.GetHausdorffDistance()

    dicecomputer = sitk.LabelOverlapMeasuresImageFilter()
    dicecomputer.Execute(labelTrue > 0.5, labelPred > 0.5)
    quality["dice"] = dicecomputer.GetDiceCoefficient()

    return quality


def evaluate_single(pred, gt):

    Recall, Specificity, Precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    MSD = []
    ASD = []
    # h = []
    b_num = pred.size(dim=0)

    for i in range(b_num):
        predict = pred[i,:]
        # pred1 = predict.data.cpu().numpy()
        pred_binary = (predict >= 0.5).float()
        # pred2 = pred_binary.data.cpu().numpy()

        pred_binary_inverse = (pred_binary == 0).float()

        gt_binary = (gt[i] >= 0.5).float()
        gt1 = gt_binary.data.cpu().numpy()
        gt_binary_inverse = (gt_binary == 0).float()

        a = pred_binary.squeeze(0).cpu()
        b = gt_binary.squeeze(0).cpu()
        # a = pred_binary.cpu()
        # b = gt_binary.cpu()
        # a_points = np.argwhere(a == 1)
        # b_points = np.argwhere(b == 1)
        # MSD.append(max(directed_hausdorff(a_points, b_points)[0], directed_hausdorff(b_points, a_points)[0]))
        MSD.append(max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0]))


        if a.max()>0 and b.max()>0:
            ASD.append(metric.binary.asd(np.array(a), np.array(b)))
        # if a.max() == 0 and b.max() == 0:
        #     ASD.append(0.0)
        # h.append(metric.binary.hd95(np.array(a), np.array(b)))
            # dict = computeQualityMeasures(a,b)
            # HD.append(max(directed_hausdorff(a,b)[0],directed_hausdorff(b,a)[0]))
            # HD.append(dict["Hausdorff"])

        TP = pred_binary.mul(gt_binary).sum()
        FP = pred_binary.mul(gt_binary_inverse).sum()
        TN = pred_binary_inverse.mul(gt_binary_inverse).sum()
        FN = pred_binary_inverse.mul(gt_binary).sum()

        if TP.item() == 0:
            # print('TP=0 now!')
            # print('Epoch: {}'.format(epoch))
            # print('i_batch: {}'.format(i_batch))
            TP = torch.Tensor([1]).cuda()

        # recall
        recall = TP / (TP + FN)

        # Specificity or true negative rate
        specificity = TN / (TN + FP)

        # Precision or positive predictive value
        precision = TP / (TP + FP)

        # F1 score = Dice
        f1 = 2 * precision * recall / (precision + recall)


        # F2 score
        f2 = 5 * precision * recall / (4 * precision + recall)

        # Overall accuracy
        aCC_overall = (TP + TN) / (TP + FP + FN + TN)

        # IoU for poly
        ioU_poly = TP / (TP + FP + FN)

        # IoU for background
        ioU_bg = TN / (TN + FP + FN)

        # mean IoU
        ioU_mean = (ioU_poly + ioU_bg) / 2.0

        Recall = Recall + recall
        Specificity = Specificity + specificity
        Precision = Precision + precision
        F1 = F1 + f1
        F2 = F2 + f2
        ACC_overall = ACC_overall + aCC_overall
        IoU_poly = IoU_poly + ioU_poly
        IoU_bg = IoU_bg + ioU_bg
        IoU_mean = IoU_mean + ioU_mean


    return Recall/b_num, Specificity/b_num, Precision/b_num, F1/b_num, F2/b_num, ACC_overall/b_num, IoU_poly/b_num, IoU_bg/b_num, IoU_mean/b_num, sum(MSD) / len(MSD), sum(ASD) / len(ASD)


class Metrics(object):
    def __init__(self, metrics_list):
        self.metrics = {}
        for metric in metrics_list:
            self.metrics[metric] = 0

    def update(self, **kwargs):
        for k, v in kwargs.items():
            assert (k in self.metrics.keys()), "The k {} is not in metrics".format(k)
            if isinstance(v, torch.Tensor):
                v = v.item()

            self.metrics[k] += v

    def mean(self, total):
        mean_metrics = {}
        for k, v in self.metrics.items():
            mean_metrics[k] = v / total
        return mean_metrics
