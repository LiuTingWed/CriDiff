import numpy as np
import SimpleITK as sitk
import os
from scipy.ndimage import zoom
from skimage import io
from skimage.util import img_as_float
from PIL import Image

def binary_loader(path):
    assert os.path.exists(path), f"`{path}` does not exist."
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("L")

def normalize_pil(pre, gt):
    gt = np.asarray(gt)
    pre = np.asarray(pre)
    gt = gt / (gt.max() + 1e-8)
    gt = np.where(gt > 0.2, 1, 0)
    pre = pre / 255

    # max_pre = pre.max()
    # min_pre = pre.min()
    # if max_pre == min_pre:
    #     pre = pre / 255
    # else:
    #     pre = (pre - min_pre) / (max_pre - min_pre)
    return pre, gt


def resize_image(img, new_shape):
    factors = [float(new_dim) / old_dim for new_dim, old_dim in zip(new_shape, img.shape)]
    resized_img = zoom(img, zoom=factors, order=1)
    return resized_img

def dice_coefficient(gt, pred):
    intersection = np.sum(gt * pred)
    union = np.sum(gt) + np.sum(pred)
    dice = (2.0 * intersection) / (union + 1e-8)
    return dice

def mean_absolute_error(gt, pred):
    mae = np.mean(np.abs(pred - gt))
    return mae

from metrics import evaluate,evaluate_single
import torch
def sort_key(file_name):
    return int(file_name.replace('.png',''))
def main(ground_truth_folder, prediction_folder, results_file):
    total_dice = 0
    total_mae = 0
    num_images = 0

    out_pred_final = torch.FloatTensor().cuda()

    out_gt = torch.FloatTensor().cuda()  # Tensor stores groundtruth values
    for file_name in enumerate(os.listdir(ground_truth_folder)):
    # for file_name in sorted(os.listdir(ground_truth_folder), key=sort_key):
        ground_truth_path = os.path.join(ground_truth_folder, file_name[1])
        #VT1000 GT .jpg not png
        file_name = file_name[1].split(".")[0]+".png"
        # file_name = file_name.split(".")[0] + '.jpg'
        prediction_path = os.path.join(prediction_folder, file_name)


        if os.path.exists(prediction_path):
            ground_truth = binary_loader(ground_truth_path)
            prediction = binary_loader(prediction_path)
            #
            # ground_truth[ground_truth < 0.5] = 0
            # ground_truth[ground_truth > 0.5] = 1
            if file_name == '23.png':
                print(1)
            if prediction.size != ground_truth.size:
                # prediction = prediction.resize(ground_truth.size, Image.BILINEAR)
                ground_truth = ground_truth.resize(prediction.size, Image.BILINEAR)

            prediction, ground_truth = normalize_pil(pre=prediction, gt=ground_truth)
            # ground_truth1 = ground_truth.data.cpu().numpy()

            dice = dice_coefficient(ground_truth, prediction)
            mae = mean_absolute_error(ground_truth, prediction)


            # ground_truth[ground_truth < 0.8] = 0

            pmean = np.mean(prediction)
            if pmean < 0.005:
                prediction[prediction < 0.5] = 0
                # prediction[prediction > 0.2] = 0

            # gmean = np.mean(ground_truth)

            ground_truth = torch.from_numpy(ground_truth).cuda()
            prediction = torch.from_numpy(prediction).cuda()


            _recallC, _specificityC, _precisionC, _F1C, _F2C, _ACC_overallC, _IoU_polyC, _IoU_bgC, _IoU_meanC, _MSD, _ASD = evaluate_single(
                prediction.unsqueeze(dim=0), ground_truth.unsqueeze(dim=0))

            print(f"Processed: {file_name}")
            print('FG_Dice %f', _F1C)
            print('score_iou_poly %f', _IoU_polyC)
            print('score_avg_msd %f', _MSD)

            # if _F1C <0.001:
            #     pass
            # else:
            #     out_pred_final = torch.cat((out_pred_final, prediction.unsqueeze(dim=0)), 0)
            #     out_gt = torch.cat((out_gt, ground_truth.unsqueeze(dim=0)), 0)
            #     num_images += 1

            out_pred_final = torch.cat((out_pred_final, prediction.unsqueeze(dim=0)), 0)
            out_gt = torch.cat((out_gt, ground_truth.unsqueeze(dim=0)), 0)
            num_images += 1

            # print(f"Processed: {file_name}")
            # print(mae)

    _recallC, _specificityC, _precisionC, _F1C, _F2C, _ACC_overallC, _IoU_polyC, _IoU_bgC, _IoU_meanC, _MSD, _ASD = evaluate_single(
        out_pred_final, out_gt)

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

    print('FG_Dice %f', FG_Dice)
    print('score_iou_poly %f', score_iou_poly)
    print('score_avg_msd %f', score_avg_msd)
    print('score_avg_asd %f', score_avg_asd)

    # average_dice = total_dice / num_images
    # average_mae = total_mae / num_images

    mode = "a" if os.path.exists(results_file) else "w"
    with open(results_file, mode) as f:
        if mode == "w":
            f.write("Ground Truth Path\tPrediction Path\tNumber of Images\tAverage Dice Coefficient\tAverage IOU\tAverage MSD\tAverage ASD\n")
        f.write(f"{ground_truth_folder}\t{prediction_folder}\t{num_images}\t{FG_Dice:.4f}\t{score_iou_poly:.4f}\t{score_avg_msd:.4f}\t{score_avg_asd:.4f}\n")
        print(f"{ground_truth_folder}\t{prediction_folder}\t{num_images}\t{FG_Dice:.4f}\t{score_iou_poly:.4f}\t{score_avg_msd:.4f}\t{score_avg_asd:.4f}\n")

if __name__ == "__main__":
    ground_truth_folder = "/home/oip/data/ProstateSeg/MUPS/masks/test"
    prediction_folder = "./state_dict_epoch_730_step_131400_dice_0.pt"
    results_file = "results.txt"
    main(ground_truth_folder, prediction_folder, results_file)