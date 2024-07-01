import torch
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import click

import argparse
import dataset
import os
import logging
import sys

import cv2
import torch.nn.functional as F
import dataset
import time
import utils
# from metric import *
from skimage import img_as_ubyte
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import SequentialSampler
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ['CUDA_VISIBLE_DEVICES']='0,1'
parser = argparse.ArgumentParser("Diffusion infer")
# Required
parser.add_argument('--cp_save', type=str, default='/media/ubuntun/file/ltw/ProstateDiffV3_MultiGPU', help='experiment name')
parser.add_argument("--loadDir", type=str, default='./exp/20240221-133240_5e-05_linear_PVT_GLenhanceV21_Diff_dim64_Prostate_init/', help="Location of the models to load in.")
parser.add_argument("--loadFile", type=str, default='/media/oip/file/ltw2/20240221-133240_5e-05_linear_PVT_GLenhanceV21_Diff_dim64_Prostate_init/state_dict_epoch_645_step_81915_dice_0.pt', help="Name of the .pkl model file to load in. Ex: model_358e_450000s.pkl")
# parser.add_argument("--loadDefFile", type=str, default=10, help="Name of the .json model file to load in. Ex: model_params_358e_450000s.pkl", required=True)
# parser.add_argument("--loadDefFile", type=str, default=10, help="Name of the .json model file to load in. Ex: model_params_358e_450000s.pkl", required=True)
parser.add_argument("--beta_sched", type=str, default='linear', help='cosine or linear')
parser.add_argument("--num_timesteps", type=float, default=500, help="Name of the .json model file to load in. Ex: model_params_358e_450000s.pkl")
parser.add_argument('--size', type=int, default=256, help='test_size')
parser.add_argument('--dataset_root', type=str, default='/home/oip/data/ProstateSeg/MUPS', help='note for this run')
parser.add_argument('--batch_size', type=int, default=5, help='batch size')
parser.add_argument('--num_ens', type=int, default=25,
                    help='number of times to sample to make an ensable of predictions like in the paper (default: 5)')

parser.add_argument('--sampling_timesteps', type=int, default=100,
                    help='number of times to sample to make an ensable of predictions like in the paper (default: 5)')
parser.add_argument('--dataset_name', type=str, default='Prostate', help='test_size')
parser.add_argument("--print_freq", type=int, default=50, help="True to put a limit on generation, False to not put a litmit on generation. If the model is generating images of a single color, then you may need to set this flag to True. Note: This restriction is usually needed when generating long sequences (low step size) Note: With a higher guidance w, the correction usually messes up generation.", required=False)

# Generation parameters
# parser.add_argument("--step_size", type=int, default=10, help="Step size when generating. A step size of 10 with a model trained on 1000 steps takes 100 steps to generate. Lower is faster, but produces lower quality images.", required=False)
parser.add_argument("--device", type=str, default="gpu", help="Device to put the model on. use \"gpu\" or \"cpu\".", required=False)
parser.add_argument("--corrected", type=bool, default=False, help="True to put a limit on generation, False to not put a litmit on generation. If the model is generating images of a single color, then you may need to set this flag to True. Note: This restriction is usually needed when generating long sequences (low step size) Note: With a higher guidance w, the correction usually messes up generation.", required=False)
parser.add_argument("--self_condition", type=bool, default=True, help="self_condition", required=False)

# Output parameters
parser.add_argument("--isGif", type=bool, default=True, help="Name of the file to save the output image to.", required=False)
parser.add_argument("--gif_fps", type=int, default=10, help="FPS for the output gif.", required=False)
parser.add_argument('--job_name', type=str, default='bs5_ddpm', help='job_name')

args, unparsed = parser.parse_known_args()
# args.job_name = 'DDIM_scale:' + str(args.DDIM_scale) + '_' + 'step_size:' + str(args.step_size)
args.job_name = 'sampling_timesteps:' + str(args.sampling_timesteps) +'_' + 'beta_sched:' + str(args.beta_sched)
args.job_name = args.loadDir + '/results/' + time.strftime("%Y%m%d-%H%M%S-")+ str(args.job_name)
utils.create_exp_dir(args.job_name)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.job_name, 'infer_log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.info("Infer exp=%s", args.loadDir)
def normalize_to_neg_one_to_one(img):
    return img * 2 - 1
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5
def mv(a: object) -> object:
    # res = Image.fromarray(np.uint8(img_list[0] / 2 + img_list[1] / 2 ))
    # res.show()
    b = a.size(0)
    return torch.sum(a, 0, keepdim=True) / b
def staple(a):
    # a: n,c,h,w detach tensor
    mvres = mv(a)
    gap = 0.4
    if gap > 0.02:
        for i, s in enumerate(a):
            r = s * mvres
            res = r if i == 0 else torch.cat((res,r),0)
        nres = mv(res)
        gap = torch.mean(torch.abs(mvres - nres))
        mvres = nres
        a = res
    return mvres
# sys.path.append('./')

# from Prostate import ProstateDataset
# from Prostate import LoadProstateDataset
from metrics import evaluate_single
from monai.utils import set_determinism

def main():
    logging.info(args)
    from monai.utils import set_determinism
    set_determinism(1)
    torch.cuda.set_device(0)  # 设置使用第0块GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 如果有GPU则使用第一块，否则使用CPU

    args.job_name = 'sampling_timesteps:' + str(args.sampling_timesteps) + '_' + 'beta_sched:' + str(
        args.beta_sched) + '_' + 'num_ens:' + str(args.num_ens)
    args.job_name = args.loadDir + '/results/' + time.strftime("%Y%m%d-%H%M%S-") + str(args.job_name)
    utils.create_exp_dir(args.job_name)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.job_name, 'infer_log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info("Infer exp=%s", args.loadDir)


    # train_data, test_data = LoadProstateDataset(CONFIG)
    # test_dataset = ProstateDataset(config=CONFIG, data=test_data, mode="test", normalization=True,
    #                                augmentation=False)
    # train_dataset = ProstateDataset(config=CONFIG, data=train_data, mode="train", normalization=True,
    #                                 augmentation=CONFIG["train"]["image_data_augmentation"])
    from Prostate_dataset import Dataset
    train_dataset = Dataset(args.dataset_root, args.size, 'train', convert_image_to='L')
    test_dataset = Dataset(args.dataset_root, args.size, 'test', convert_image_to='L')



    from module.DiffusionModel import DiffSOD
    # Create a dummy model


    from skimage import io
    # weight_map = io.imread('./train_256.png')
    # weight_map_tensor = torch.from_numpy(weight_map).float()
    # # weight_map_tensor = weight_map_tensor > 155
    # # weight_map_tensor.cuda(local_rank)
    # # 将张量限制在 0-1 范围内
    # weight_map_tensor = torch.clamp(weight_map_tensor, 0, 1)

    # 计算最小值和最大值
    # min_value = torch.min(weight_map_tensor)
    # max_value = torch.max(weight_map_tensor)

    # 归一化张量到 0-1 范围
    # weight_map_tensor = (weight_map_tensor - min_value) / (max_value - min_value)

    # diffusion = DiffSOD(args, sampling_timesteps=args.sampling_timesteps if args.sampling_timesteps > 0 else None)
    diffusion = DiffSOD(args, sampling_timesteps=args.sampling_timesteps if args.sampling_timesteps > 0 else None)
    diffusion = diffusion.to(device)
    # from collections import OrderedDict
    # if args.loadFile is not None:
    #     # 加载保存的状态字典
    #     save_dict = torch.load(os.path.join(args.cp_save, args.loadFile),
    #                            map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
    #
    #     # 创建一个新字典，其键没有'module.'前缀
    #     new_state_dict = OrderedDict()
    #     for k, v in save_dict['model_state_dict'].items():
    #         name = k[7:] if k.startswith('module.') else k  # 删除'module.'前缀
    #         new_state_dict[name] = v
    #
    #     # 加载状态字典
    #     diffusion.model.load_state_dict(new_state_dict, strict=True)
    save_dict = torch.load(args.loadFile)
    diffusion.load_state_dict(save_dict['model_state_dict'],strict=True)
    # img_root = args.dataset_root + dataset_name

    # test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size,
    #                                               num_workers=0, pin_memory=True, sampler=DistributedSampler(test_dataset, shuffle=False, drop_last=False))
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=args.batch_size,
                                                  # num_workers=0,  # 设置为CPU核心数，以提高数据加载速度
                                                  pin_memory=True,
                                                  shuffle=False,  # 可以根据需要设定是否打乱数据
                                                  drop_last=False)  # 是否丢弃最后一个不完整的batch

    num_test = len(test_dataset)
    score_metricsC, score_metricsT = epoch_evaluating(diffusion, test_dataloader, device, evaluate_single, device,
                                                      last_Dicescore=0)
    FG_Dice = score_metricsC["f1"].item()  # baseline 80 sota 86
    score_f2 = score_metricsC["f2"].item()
    score_precision = score_metricsC["precision"].item()
    score_accuracy = score_metricsC["accuracy"].item()
    score_recall = score_metricsC["recall"].item()
    score_iou_poly = score_metricsC["iou_poly"].item()
    score_iou_bg = score_metricsC["iou_bg"].item()
    score_iou_mean = score_metricsC["iou_mean"].item()  # baseline 78 sota 82
    score_avg_msd = score_metricsC["avg_msd"].item()  # baseline 2.16 sota 1.96
    score_avg_asd = score_metricsC["avg_asd"].item()  # baseline 2.16 sota 1.96

    logging.info('dataset_name {}'.format(args.dataset_name))
    logging.info('FG_Dice %f', FG_Dice)
    # logging.info('score_f2 %f', score_f2)
    logging.info('Iou_poly %f', score_iou_poly)
    logging.info('Iou_mean %f', score_iou_mean)
    # logging.info('score_accuracy %f', score_accuracy)
    logging.info('MSD %f', score_avg_msd)
    logging.info('ASD %f', score_avg_asd)


from medpy import metric
import skimage.io as io
# Define training evaluation for each epoch
import numpy as np


class Dice(object):
    def __init__(self):
        self.dices = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        dice = self.cal_dice(pred, gt)
        self.dices.append(dice)

    def cal_dice(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        smooth = 1e-5
        gt_f = gt.flatten()
        pred_f = pred.flatten()
        intersection = np.sum(gt_f * pred_f)
        dice = (2. * intersection + smooth) / (np.sum(gt_f) + np.sum(pred_f) + smooth)
        return dice

    def get_results(self) -> dict:
        dice = np.mean(np.array(self.dices, ))
        return dict(dice=dice)

def mv(a: object) -> object:
    # res = Image.fromarray(np.uint8(img_list[0] / 2 + img_list[1] / 2 ))
    # res.show()
    b = a.size(0)
    return torch.sum(a, 0, keepdim=True) / b
def staple(a):
    # a: n,c,h,w detach tensor
    mvres = mv(a)
    gap = 0.4
    if gap > 0.02:
        for i, s in enumerate(a):
            r = s * mvres
            res = r if i == 0 else torch.cat((res,r),0)
        nres = mv(res)
        gap = torch.mean(torch.abs(mvres - nres))
        mvres = nres
        a = res
    return mvres

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()
import torchvision.utils as vutils
def epoch_evaluating(model, test_dataloader, device, criteria_metrics, local_rank, last_Dicescore):
    # Switch model to evaluation mode
    # set_determinism(1)

    model.eval()
    out_pred_final = torch.FloatTensor().cuda(local_rank)
    out_gt = torch.FloatTensor().cuda(local_rank)  # Tensor stores groundtruth values
    savepath = 'prediction/' + args.dataset_name
    with torch.no_grad():  # Turn off gradient
        # For each batch
        test_output_root = os.path.join(args.job_name, savepath)

        for step, (images, masks, index) in enumerate(test_dataloader):
            # Move images, labels to device (GPU)
            input = images.cuda(local_rank)
            masks = masks.cuda(local_rank)
            # input = images * 2 - 1
            preds = torch.zeros((input.shape[0], args.num_ens, input.shape[2], input.shape[3])).cuda(local_rank)
            # preds = []
            for i in range(args.num_ens):

                # pred_noise, pred_cal = model.sample(input)
                pred_noise, pred_gif= model.sample(input)

                # for j in range(input.shape[0]):
                #     _recallC, _specificityC, _precisionC, _F1C, _F2C, _ACC_overallC, _IoU_polyC, _IoU_bgC, _IoU_meanC, _MSD, _ASD = criteria_metrics(
                #         pred_cal[j].sigmoid(), masks[j])
                #     logging.info('pred_cal')
                #     logging.info('ID {}'.format(str(index[j])))
                #     logging.info('ens {}'.format(i))
                #     logging.info('FG_Dice %f', _F1C)
                #     logging.info('Iou_poly %f', _IoU_polyC)
                #     logging.info('MSD %f', _MSD)
                #     logging.info('ASD %f', _ASD)
                #     # preds_mean = staple(torch.stack(preds, dim=0)).squeeze(0)
                #
                #     predict_cal = img_as_ubyte(pred_cal[j].sigmoid().cpu().numpy())
                #     if not os.path.exists(test_output_root):
                #         os.makedirs(test_output_root)
                #     # for i in range(input.shape[0]):
                #     cv2.imwrite(test_output_root + '/' + str(index[j]) + '_cal_' + str(i) +'.png', predict_cal.squeeze())

                # _recallC, _specificityC, _precisionC, _F1C, _F2C, _ACC_overallC, _IoU_polyC, _IoU_bgC, _IoU_meanC, _MSD, _ASD = criteria_metrics(
                #     pred_noise, pred_cal.sigmoid())
                #
                # d1 = _F1C
                # if d1 < 0.65:
                #     cal_out = torch.clamp(pred_cal + 0.25 * pred_noise, 0, 1)
                # else:
                #     cal_out = torch.clamp(pred_cal * 0.5 + 0.5 * pred_noise, 0, 1)
                # preds.append(cal_out)
                preds[:, i:i + 1, :, :] = pred_noise

            # preds_mean = staple(preds).squeeze(0)

            preds_mean = preds.mean(dim=1)

            # out_pred_final = torch.cat((out_pred_final, preds_mean), 0)
            # preds_mean = preds_mean
            preds_mean[preds_mean < 0.3] = 0
            # masks = torch.squeeze(masks)
            # preds_std = preds.std(dim=1)
            for idx in range(len(pred_gif)):
                predict_rgb = pred_gif[idx].cpu().detach()
                predict_rgb = torch.squeeze(predict_rgb)
                if not os.path.exists(test_output_root):
                    os.makedirs(test_output_root)
                # for i in range(input.shape[0]):
                # cv2.imwrite(test_output_root + '/' + str(index[idx]) + '.png', predict_rgb[idx])
                vutils.save_image(predict_rgb, test_output_root + '/' + str(index[0])+'_'+ str(idx)+ '.png')
            # for idx in range(images.shape[0]):
            #     predict_rgb = preds_mean.cpu().detach()
            #     predict_rgb = torch.squeeze(predict_rgb)
            #     # predict_rgb = predict_rgb.sigmoid().numpy()
            #     # predict_rgb = (predict_rgb / predict_rgb.max()).cpu().detach().numpy()
            #     # predict_rgb = img_as_ubyte(predict_rgb)
            #     if not os.path.exists(test_output_root):
            #         os.makedirs(test_output_root)
            #     # for i in range(input.shape[0]):
            #     # cv2.imwrite(test_output_root + '/' + str(index[idx]) + '.png', predict_rgb[idx])
            #     vutils.save_image(predict_rgb[idx], test_output_root + '/' + str(index[idx]) + '.png')
            #     preds_single_eval = preds_mean[idx, :, :].unsqueeze(dim=0)
            #     _recallC, _specificityC, _precisionC, _F1C, _F2C, _ACC_overallC, _IoU_polyC, _IoU_bgC, _IoU_meanC, _MSD, _ASD = criteria_metrics(
            #         preds_single_eval, masks[idx])
            #     # logging.info()
            #     # preds_single_eval_1 = preds_single_eval.cpu().numpy()
            #     # mask_1=masks[idx].cpu().numpy()
            #     # if preds_single_eval_1.sum() != 0 and mask_1.sum() != 0:
            #     #     dice = metric.binary.dc(preds_single_eval_1, mask_1)
            #     #     iou = metric.binary.jc(preds_single_eval_1, mask_1)
            #     #     msd = metric.binary.hd95(preds_single_eval_1, mask_1)
            #     #     logging.info('ID {}'.format(str(index[idx])))
            #     #     logging.info('Dice %f', dice)
            #     #     logging.info('Iou %f', iou)
            #     #     logging.info('Msd %f', msd)
            #     # iou = metric.binary.i
            #     logging.info('ID {}'.format(str(index[idx])))
            #     logging.info('FG_Dice %f', _F1C)
            #     logging.info('Iou_poly %f', _IoU_polyC)
            #     logging.info('Iou_mean %f', _IoU_meanC)
            #     logging.info('MSD %f', _MSD)
            #     logging.info('ASD %f', _ASD)

            # preds_mean1 = preds_mean.squeeze().cpu().detach().numpy()
            #
            # target = masks.squeeze().cpu().detach().numpy()
            # cal_fm.update(preds_mean1, target)
            # fmeasure, maxf, mmf, _, _ = cal_fm.show()
            # import numpy as np
            # fmeasure = np.mean(fmeasure)

            # _recallC, _specificityC, _precisionC, _F1C, _F2C, _ACC_overallC, _IoU_polyC, _IoU_bgC, _IoU_meanC, _MSD = criteria_metrics(
            #     out_pred_final, out_gt)
            # if loss < IoU_loss_bank:
            #     save_image(pred=predicts, masks=masks, test_step=step)

            # if name[0] == 'sun_aridzcvvxhxzukzp':
            #     print(1)
            # if args.isGif == True and name[0] == 'sun_aridzcvvxhxzukzp':
            # if args.isGif == True:
            #     # Image evolution gif
            #     plt.close('all')
            #     fig, ax = plt.subplots()
            #     ax.set_axis_off()
            #     for i in range(0, len(imgs)):
            #         title = plt.text(imgs[i].shape[0] // 2, -5, f"t = {i}", ha='center')
            #         imgs[i] = [plt.imshow(imgs[i], animated=True), title]
            #     animate = animation.ArtistAnimation(fig, imgs, interval=1, blit=True, repeat_delay=1000)
            #     animate.save(test_output_root + '/DiffusionSOD.gif', writer=animation.PillowWriter(fps=args.gif_fps))
            # if not os.path.exists(test_output_root):
            #     os.makedirs(test_output_root)
            # predict_rgb[predict_rgb < 0] = 0
            # out_pred_final = torch.cat((out_pred_final, predict_rgb[0]), 0)
            # # out_pred_7 = torch.cat((out_pred_7, predictions[1][0]), 0)
            # # out_pred_14 = torch.cat((out_pred_14, predictions[1][1]), 0)
            # # out_pred_28 = torch.cat((out_pred_28, predictions[1][2]), 0)
            # # out_pred_56 = torch.cat((out_pred_56, predictions[1][3]), 0)
            # predict_rgb = torch.squeeze(predict_rgb)
            # # predict_rgb = predict_rgb.sigmoid().cpu().numpy()
            # predict_rgb = (predict_rgb / predict_rgb.max()).cpu().detach().numpy()
            # predict_rgb[predict_rgb < -1] = -1
            #
            # # predict_rgb2 = output_rgb2.sigmoid().cpu().numpy()
            # # predict_rgb3 = output_rgb3.sigmoid().cpu().numpy()
            # predict_rgb = img_as_ubyte(predict_rgb)
            #
            # # predict_rgb = predict_rgb.resize(target.size, Image.BILINEAR)
            # cv2.imwrite(test_output_root + '/' + str(index.cpu().detach().numpy()) + '.png', predict_rgb)
            #
            if local_rank == 0 and step % args.print_freq == 0 or step == len(test_dataloader) - 1:
                logging.info(
                    "val: Step {:03d}/{:03d}".format(step, len(test_dataloader) - 1))

    _recallC, _specificityC, _precisionC, _F1C, _F2C, _ACC_overallC, _IoU_polyC, _IoU_bgC, _IoU_meanC, _MSD, _ASD = criteria_metrics(
        out_pred_final, out_gt)
    # _recallT, _specificityT, _precisionT, _F1T, _F2T, _ACC_overallT, _IoU_polyT, _IoU_bgT, _IoU_meanT = criteria_metrics(
    #     out_pred_trans, out_gt)
    score_metricsC = {
        "recall": _recallC,
        "specificity": _specificityC,
        "precision": _precisionC,
        "f1": _F1C,
        "f2": _F2C,
        "accuracy": _ACC_overallC,
        "iou_poly": _IoU_polyC,
        "iou_bg": _IoU_bgC,
        "iou_mean": _IoU_meanC,
        "avg_msd": _MSD,
        "avg_asd": _ASD,
    }
    # if _F1C > last_Dicescore:
    #     save_image(out_pred_final, out_gt, name='ProstateX_UnetPlus_train')
    #     # save_image(out_pred_7, out_gt, name='Pred_my_7')
    #     # save_image(out_pred_14, out_gt, name='Pred_my_14')
    #     # save_image(out_pred_28, out_gt, name='Pred_my_28')
    #     # save_image(out_pred_56, out_gt, name='Pred_my_56')
    score_metricsT = {
        # "recall": _recallT,
        # "specificity": _specificityT,
        # "precision": _precisionT,
        # "f1": _F1T,
        # "f2": _F2T,
        # "accuracy": _ACC_overallT,
        # "iou_poly": _IoU_polyT,
        # "iou_bg": _IoU_bgT,
        # "iou_mean": _IoU_meanT
    }

    # Clear memory
    # del images, masks, out_pred_final, out_gt, out_pred_7, out_pred_14, out_pred_28, out_pred_56
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # return validation loss, and metric score
    return score_metricsC, score_metricsT


if __name__ == '__main__':
    main()
