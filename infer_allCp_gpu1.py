import torch
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import click

import argparse
import os
import logging
import sys

import cv2
import torch.nn.functional as F
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
parser.add_argument("--loadDir", type=str, default='./exp/20240221-133240_5e-05_linear_PVT_GLenhanceV21_Diff_dim64_Prostate_init', help="Location of the models to load in.")
parser.add_argument("--loadDer_cp", type=str, default='/media/oip/file/ltw2/20240221-133240_5e-05_linear_PVT_GLenhanceV21_Diff_dim64_Prostate_init/', help="Name of the .json model file to load in. Ex: model_params_358e_450000s.pkl")
parser.add_argument("--beta_sched", type=str, default='linear', help='cosine or linear')
parser.add_argument("--num_timesteps", type=float, default=500, help="Name of the .json model file to load in. Ex: model_params_358e_450000s.pkl")
parser.add_argument('--size', type=int, default=256, help='test_size')
parser.add_argument('--dataset_root', type=str, default='/home/oip/data/ProstateSeg/PROMISE12', help='note for this run')
parser.add_argument('--batch_size', type=int, default=5, help='batch size')
parser.add_argument('--num_ens', type=int, default=25,
                    help='number of times to sample to make an ensable of predictions like in the paper (default: 5)')

parser.add_argument('--sampling_timesteps', type=int, default=30,
                    help='number of times to sample to make an ensable of predictions like in the paper (default: 5)')
parser.add_argument('--dataset_name', type=str, default='Prostate158', help='test_size')
parser.add_argument("--print_freq", type=int, default=50, help="True to put a limit on generation, False to not put a litmit on generation. If the model is generating images of a single color, then you may need to set this flag to True. Note: This restriction is usually needed when generating long sequences (low step size) Note: With a higher guidance w, the correction usually messes up generation.", required=False)

# Generation parameters
# parser.add_argument("--step_size", type=int, default=10, help="Step size when generating. A step size of 10 with a model trained on 1000 steps takes 100 steps to generate. Lower is faster, but produces lower quality images.", required=False)
parser.add_argument("--device", type=str, default="gpu", help="Device to put the model on. use \"gpu\" or \"cpu\".", required=False)
parser.add_argument("--corrected", type=bool, default=False, help="True to put a limit on generation, False to not put a litmit on generation. If the model is generating images of a single color, then you may need to set this flag to True. Note: This restriction is usually needed when generating long sequences (low step size) Note: With a higher guidance w, the correction usually messes up generation.", required=False)
parser.add_argument("--self_condition", type=bool, default=True, help="self_condition", required=False)

# Output parameters
parser.add_argument("--isGif", type=bool, default=True, help="Name of the file to save the output image to.", required=False)
parser.add_argument("--gif_fps", type=int, default=10, help="FPS for the output gif.", required=False)
parser.add_argument('--job_name', type=str, default='B1S-1DDPM', help='job_name')

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

# from Prostate import ProstateDataset
# from Prostate import LoadProstateDataset
from metrics import evaluate_single
def main():
    from monai.utils import set_determinism
    set_determinism(1)
    logging.info(args)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(1)  # 设置使用第0块GPU
    args.job_name = 'sampling_timesteps:' + str(args.sampling_timesteps) + '_' + 'beta_sched:' + str(
        args.beta_sched)
    args.job_name = args.loadDir + '/results/' + time.strftime("%Y%m%d-%H%M%S-") + str(args.job_name)
    utils.create_exp_dir(args.job_name)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.job_name, 'infer_log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info("Infer exp=%s", args.loadDir)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(args.job_name, "tb"))


    from module.DiffusionModel import DiffSOD

    from Prostate_dataset import Dataset
    # train_dataset = Dataset(args.dataset_root, args.size, 'train', convert_image_to='L')
    test_dataset = Dataset(args.dataset_root, args.size, 'test', convert_image_to='L')
    # batchSize = args.batch_size // args.numSteps

    # sample_batch_size = args.sample_batch_size // args.numSteps

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=0,
        pin_memory=True, shuffle=False)

    diffusion = DiffSOD(args, sampling_timesteps=args.sampling_timesteps if args.sampling_timesteps > 0 else None)
    diffusion = diffusion.to(device)


    # 获取检查点文件列表
    checkpoint_dir = args.loadDer_cp
    # checkpoint_files = sorted(os.listdir(checkpoint_dir))
    # checkpoint_files = [f for f in sorted(os.listdir(checkpoint_dir)) if f.endswith('.pt')]
    # num_checkpoints = len(checkpoint_files)
    unsorted_checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    # 定义一个函数来提取文件名中的数字
    import re
    def extract_number(f):
        s = re.findall(r'\d+', f)
        return (int(s[0]) if s else -1, f)
    # 根据提取的数字对文件列表进行排序
    checkpoint_files = sorted(unsorted_checkpoint_files, key=extract_number, reverse=True)
    num_checkpoints = len(checkpoint_files)

    # 前一半的检查点
    checkpoints_to_validate = checkpoint_files[int(num_checkpoints/2):]

    # 循环验证分配给这个GPU的检查点
    from collections import OrderedDict

    for checkpoint_file in checkpoints_to_validate:
        epoch = checkpoint_file.split("_")[3]
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
        save_dict = torch.load(checkpoint_path)
        new_state_dict = OrderedDict()
        for k, v in save_dict['model_state_dict'].items():
            name = k[7:] if k.startswith('module.') else k  # 删除'module.'前缀
            new_state_dict[name] = v

        # 加载状态字典
        diffusion.load_state_dict(new_state_dict, strict=True)

        logging.info('CP_name {}'.format(checkpoint_file))
        score_metricsC, score_metricsT = epoch_evaluating(diffusion, checkpoint_file, test_dataloader, device, evaluate_single)
        FG_Dice = score_metricsC["f1"].item() # baseline 80 sota 86
        score_iou_poly = score_metricsC["iou_poly"].item()
        # score_iou_mean = score_metricsC["iou_mean"].item() # baseline 78 sota 82
        score_avg_msd = score_metricsC["avg_msd"].item()  # baseline 2.16 sota 1.96
        score_avg_asd = score_metricsC["avg_asd"].item()  # baseline 2.16 sota 1.96

        logging.info('dataset_name {}'.format(args.dataset_name))
        logging.info('FG_Dice %f', FG_Dice)
        # logging.info('score_f2 %f', score_f2)
        logging.info('Iou_poly %f', score_iou_poly)
        # logging.info('Iou_mean %f', score_iou_mean)
        # logging.info('score_accuracy %f', score_accuracy)
        logging.info('MSD %f', score_avg_msd)
        logging.info('ASD %f', score_avg_asd)

        writer.add_scalars('Metrics', {'FG_Dice_mean': FG_Dice * 10,
                                       'Iou_mean': score_iou_poly * 10,
                                       'Avg_msd': score_avg_msd,
                                       'Avg_asd': score_avg_asd},
                           (int(epoch)))


from medpy import metric
import skimage.io as io
# Define training evaluation for each epoch
# import numpy as np
# class Dice(object):
#     def __init__(self):
#         self.dices = []
#
#     def step(self, pred: np.ndarray, gt: np.ndarray):
#
#         dice = self.cal_dice(pred, gt)
#         self.dices.append(dice)
#
#     def cal_dice(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
#         smooth = 1e-5
#         gt_f = gt.flatten()
#         pred_f = pred.flatten()
#         intersection = np.sum(gt_f * pred_f)
#         dice = (2. * intersection + smooth) / (np.sum(gt_f) + np.sum(pred_f) + smooth)
#         return dice
#
#     def get_results(self) -> dict:
#         dice = np.mean(np.array(self.dices, ))
#         return dict(dice=dice)

def epoch_evaluating(model, checkpoint_file, test_dataloader, device, criteria_metrics):
    # Switch model to evaluation mode
    model.eval()
    out_pred_final = torch.FloatTensor().cuda(device)
    out_gt = torch.FloatTensor().cuda(device)  # Tensor stores groundtruth values
    savepath = './prediction/' + args.dataset_name + "/" + checkpoint_file
    with torch.no_grad():  # Turn off gradient
        # For each batch
        test_output_root = os.path.join(args.job_name, savepath)

        for step, (images, masks, index) in enumerate(test_dataloader):
            # Move images, labels to device (GPU)
            input = images.cuda(device)
            masks = masks.cuda(device)
            # input = images * 2 - 1
            preds = torch.zeros((input.shape[0], args.num_ens, input.shape[2], input.shape[3])).cuda(device)
            for i in range(args.num_ens):
                preds[:, i:i + 1, :, :] = model.sample(input)
            preds_mean = preds.mean(dim=1)
            # out_pred_final = torch.cat((out_pred_final, preds_mean), 0)
            # preds_mean = preds_mean
            preds_mean[preds_mean < 0.3] = 0
            # preds_mean[preds_mean > 1] = 1

            # preds_mean1 = preds_mean.data.cpu().numpy()
            out_pred_final = torch.cat((out_pred_final, preds_mean), 0)
            out_gt = torch.cat((out_gt, masks), 0)
            # masks = torch.squeeze(masks)
            # preds_std = preds.std(dim=1)
            for idx in range(preds.shape[0]):
                predict_rgb = preds_mean[idx, :, :].cpu().detach()
                # # predict_rgb = torch.squeeze(predict_rgb)
                # predict_rgb = predict_rgb.sigmoid().numpy()
                # predict_rgb = (predict_rgb / predict_rgb.max()).cpu().detach().numpy()
                predict_rgb = img_as_ubyte(predict_rgb)
                if not os.path.exists(test_output_root):
                    os.makedirs(test_output_root)
                # for i in range(input.shape[0]):
                cv2.imwrite(test_output_root + '/' + str(index[idx]) + '.png', predict_rgb)

                # preds_single_eval = preds_mean[idx, :, :].unsqueeze(dim=0)
                # _recallC, _specificityC, _precisionC, _F1C, _F2C, _ACC_overallC, _IoU_polyC, _IoU_bgC, _IoU_meanC, _MSD, _ASD = criteria_metrics(
                #     preds_single_eval, masks[idx])
                # logging.info()
                # preds_single_eval_1 = preds_single_eval.cpu().numpy()
                # mask_1=masks[idx].cpu().numpy()
                # if preds_single_eval_1.sum() != 0 and mask_1.sum() != 0:
                #     dice = metric.binary.dc(preds_single_eval_1, mask_1)
                #     iou = metric.binary.jc(preds_single_eval_1, mask_1)
                #     msd = metric.binary.hd95(preds_single_eval_1, mask_1)
                #     logging.info('ID {}'.format(str(index[idx])))
                #     logging.info('Dice %f', dice)
                #     logging.info('Iou %f', iou)
                #     logging.info('Msd %f', msd)
                # iou = metric.binary.i
                # logging.info('ID {}'.format(str(index[idx])))
                # logging.info('FG_Dice %f', _F1C)
                # logging.info('Iou_poly %f', _IoU_polyC)
                # logging.info('Iou_mean %f', _IoU_meanC)
                # logging.info('MSD %f', _MSD)
                # logging.info('ASD %f', _ASD)
            if step % args.print_freq == 0 or step == len(test_dataloader) - 1:
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
        "avg_msd":_MSD,
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
