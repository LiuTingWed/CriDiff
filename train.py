import sys
import numpy as np
import time
import torch
import utils
import glob
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
parser = argparse.ArgumentParser("Diffusion")
##training setting
parser.add_argument('--dataset_name', type=str, default='PROMISE12', help='test_size')
parser.add_argument('--dataset_root', type=str, default='/home/oip/data/ProstateSeg/PROMISE12', help='dataset_root')
parser.add_argument('--checkpoint_dir', type=str, default='/media/oip/file/ltw2', help='other large space path to save ck')
parser.add_argument('--save', type=str, default='./exp', help='experiment name')

parser.add_argument('--batch_size', type=int, default=6, help='batch size')
parser.add_argument('--epochs', type=int, default=2000, help='num of training epochs')
parser.add_argument('--size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.00005, help='init learning rate')
parser.add_argument('--momentum', type=float, default= 0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--num_timesteps', type=int, default=500, help='batch size')

parser.add_argument('--self_condition', type=bool, default=True, help='cosine or liner')
parser.add_argument('--beta_sched', type=str, default='linear', help='cosine or liner')
parser.add_argument('--numSteps', type=int, default=1, help='Number of steps to breakup the batchSize into.')
parser.add_argument('--sample_batch_size', type=int, default=5, help='batch size')
parser.add_argument('--num_ens', type=int, default=1, help='batch size')
parser.add_argument('--sampling_timesteps', type=int, default=30, help='batch size')

parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--print_freq', type=int, default=50, help='note for this run')
parser.add_argument('--job_name', type=str, default='PVT_GLenhanceV21_Diff_dim64_Prostate_init', help='job_name')

args, unparsed = parser.parse_known_args()

# args.save = '{}-lr:{}-'.format(time.strftime("%Y%m%d-%H%M%S"), args.learning_rate)


def init_distributed():
    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    dist_url = "env://"  # default

    # only works with torch.distributed.launch // torchrun
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    # Try the nccl backend
    try:
        dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)
    # Use the gloo backend if nccl isn't supported
    except RuntimeError:
        dist.init_process_group(
            backend="gloo",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)
    # synchronizes all the threads to reach this point before moving on
    dist.barrier()

def main():
    from monai.utils import set_determinism
    # set_determinism(233)
    init_distributed()
    max_world_size = None
    device = "gpu"
    if device.lower() == "gpu":
        if torch.cuda.is_available():
            dev = device.lower()
            local_rank = int(os.environ['LOCAL_RANK']) if max_world_size is None else min(int(os.environ['LOCAL_RANK']),
                                                                                          max_world_size)
            device = torch.device(f"cuda:{local_rank}")
        else:
            dev = "cpu"
            print("GPU not available, defaulting to CPU. Please ignore this message if you do not wish to use a GPU\n")
            device = torch.device('cpu')
    else:
        dev = "cpu"
        device = torch.device('cpu')
        raise TypeError


    if args.job_name != '':
        args.job_name = time.strftime("%Y%m%d-%H%M%S_") + str(args.learning_rate) +'_'+ args.beta_sched +'_'+ args.job_name
        args.save = os.path.join(args.save, args.job_name)
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.job_name)
        if local_rank == 0:
            utils.create_exp_dir(args.save,scripts_to_save=glob.glob('*.py'))
            os.system('cp -r ./module '+args.save)
            log_format = '%(asctime)s %(message)s'
            logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                                format=log_format, datefmt='%m/%d %I:%M:%S %p')
            fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
            fh.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(fh)
            writer = SummaryWriter(log_dir=os.path.join(args.save, "tb"))
            # os.system(f'rsync -av --exclude="{args.save}" ./ {args.save}/')
    else:
        args.save = os.path.join(args.save, 'output')
        if local_rank == 0:
            utils.create_exp_dir(args.save)

    if local_rank == 0:
        logging.info("args = %s", args)
        logging.info("unparsed_args = %s", unparsed)



    device = device
    dev = dev
    from module.DiffusionModel import DiffSOD
    if dev != "cpu":
    # Initialize the environment
    #     save_dict = torch.load(
    #         "./exp/20231220-220824_5e-05_linear_ResUnet_GLenhanceV21_detail_body_decouple_debody_crssAtt_Prostate/state_dict_epoch_210_step_26670_dice_0_22222808003425598.pt")[
    #         'model_state_dict']
    #     save_dict = torch.load(
    #         "./generate_pharse/results/model-8.pt")[
    #         'model']
    #     filtered_state_dict = {k: v for k, v in save_dict.items() if 'model' in k}
    #     fixed_state_dict = {k.replace('model.', ''): v for k, v in filtered_state_dict.items()}
    #     fixed_state_dict2 = {}
    #     # downs_label_noise.1.2.fn.fn.to_k.weight
    #     key_ex = ["downs_label_noise.1.2.fn.fn.to_k.weight","downs_label_noise.1.2.fn.fn.to_v.weight","downs_label_noise.2.2.fn.fn.to_k.weight","downs_label_noise.2.2.fn.fn.to_v.weight","downs_label_noise.3.2.fn.fn.to_k.weight","downs_label_noise.3.2.fn.fn.to_v.weight","ups.1.2.fn.fn.to_v.weight","ups.1.2.fn.fn.to_k.weight","ups.0.2.fn.fn.to_v.weight","ups.0.2.fn.fn.to_k.weight"]
    #     for name, param in fixed_state_dict.items():
    #         a = name.split(".")[-2]
    #         if name == "downs_label_noise.1.2.fn.fn.to_k.weight":
    #             pass
    #         if name == 'downs_label_noise.1.2.fn.fn.to_v.weight':
    #             pass
    #         if name == 'downs_label_noise.2.2.fn.fn.to_k.weight':
    #             pass
    #         if name == 'downs_label_noise.2.2.fn.fn.to_v.weight':
    #             pass
    #         if name == 'downs_label_noise.3.2.fn.fn.to_k.weight':
    #             pass
    #         if name == 'downs_label_noise.3.2.fn.fn.to_v.weight':
    #             pa100ss
    #         fixed_state_dict2[name] = param
    #     fixed_state_dict2 = {k: v for k, v in fixed_state_dict.items() if k not in key_ex}

        DiffModel = DiffSOD(args, sampling_timesteps=args.sampling_timesteps if args.sampling_timesteps > 0 else None)
        # DiffModel.model.load_state_dict(fixed_state_dict2, strict=False)
        model = DDP(DiffModel.cuda(), device_ids=[local_rank], find_unused_parameters=False)

    else:
        raise ValueError

    model = model.to(device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=1e-4)
    if local_rank == 0:
        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))


    from Prostate_dataset import Dataset
    train_dataset = Dataset(args.dataset_root, args.size, 'train', convert_image_to='L')
    test_dataset = Dataset(args.dataset_root, args.size, 'test', convert_image_to='L')
    batchSize = args.batch_size // args.numSteps

    sample_batch_size = args.sample_batch_size // args.numSteps

    train_queue = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchSize,drop_last=False,
        pin_memory=True,sampler=DistributedSampler(train_dataset, shuffle=True),num_workers=0)
    test_queue = torch.utils.data.DataLoader(
        test_dataset, batch_size=sample_batch_size, num_workers=0,
        pin_memory=True, shuffle=False)


    steps_list = np.array([])

    # Number of steps taken
    num_steps = 0

    # Cumulative loss over the batch over each set of steps
    losses_comb_s = torch.tensor(0.0, requires_grad=False)
    losses_sp_s= torch.tensor(0.0, requires_grad=False)
    losses_side_s = torch.tensor(0.0, requires_grad=False)

    losses_body_s = torch.tensor(0.0, requires_grad=False)
    losses_detail_s = torch.tensor(0.0, requires_grad=False)

    from loss import  lossFunct, structure_loss, comput_loss
    numSteps = args.numSteps
    best_dice = 0

    for epoch in range(1, args.epochs + 1):
        losses_comb = np.array([])
        losses_sp = np.array([])
        losses_side = np.array([])
        losses_body = np.array([])
        losses_detail = np.array([])
        if local_rank == 0:
            logging.info('Epoch: %d', epoch)

        if dev != "cpu":
            train_queue.sampler.set_epoch(epoch)


        model.train()
        for step, (input, label, body_data, detail_data) in enumerate(train_queue):
            input = input.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            body_data = body_data.cuda(non_blocking=True)
            detail_data = detail_data.cuda(non_blocking=True)

            # Increate the number of steps taken
            num_steps += 1
            loss_simple, input_side_out, body_pre, detail_pre = model.module(input, label)
            # loss_simple = 0.0
            loss_body = comput_loss(body_pre, body_data, type = 'bce')
            loss_detail = comput_loss(detail_pre, detail_data, type = 'bce')
            side_out_loss = comput_loss(input_side_out, label, type = 'bce+iou')

            loss = loss_simple + side_out_loss + loss_body + loss_detail

            loss.backward()

            # Save the loss values
            losses_comb_s += loss.cpu().detach()
            losses_sp_s += loss_simple.cpu().detach()
            # losses_sp_s += loss_simple

            losses_side_s += side_out_loss.cpu().detach()
            losses_body_s += loss_body.cpu().detach()
            losses_detail_s += loss_detail.cpu().detach()

            if local_rank == 0:
                if step % args.print_freq == 0 or step == len(train_queue) - 1:
                        logging.info(
                            "train: [{:2d}/{}] Step {:03d}/{:03d} Loss {sal_losses:.3f} SpLoss {losses_sp:.3f} SideLoss {sideloss:.3f} BodyLoss {bodyloss:.3f} DetailLoss {detailloss:.3f}".format(
                                epoch, args.epochs, step, len(train_queue) - 1, sal_losses=losses_comb_s.cpu().detach().item(), losses_sp=losses_sp_s.cpu().detach().item() ,sideloss=losses_side_s.cpu().detach().item(), bodyloss=losses_body_s.cpu().detach().item(), detailloss=losses_detail_s.cpu().detach().item()))


                writer.add_scalar('Loss', losses_comb_s.cpu().detach().item(), ((epoch + 1) * len(train_queue) + step + 1))
                writer.add_scalar('SpLoss', losses_sp_s.cpu().detach().item(),
                                  ((epoch + 1) * len(train_queue) + step + 1))
                # writer.add_scalar('KlLoss', losses_var_s.cpu().detach().item(), ((epoch + 1) * len(train_queue) + step + 1))
                writer.add_scalar('SideLoss', losses_side_s.cpu().detach().item(),
                                  ((epoch + 1) * len(train_queue) + step + 1))
                writer.add_scalar('BodyLoss', losses_body_s.cpu().detach().item(),
                                  ((epoch + 1) * len(train_queue) + step + 1))
                writer.add_scalar('DetailLoss', losses_detail_s.cpu().detach().item(),
                                  ((epoch + 1) * len(train_queue) + step + 1))
                # writer.add_scalar('SalLoss', losses_sal_s.cpu().detach().item(),
                #                   ((epoch + 1) * len(train_queue) + step + 1))

            # If the number of steps taken is a multiple of the number
            # of desired steps, update the models
            if num_steps % numSteps == 0:
                # Update the model using all losses over the steps
                optimizer.step()
                optimizer.zero_grad()
                # scheduler.step()


                # Save the loss values
                losses_comb = np.append(losses_comb, losses_comb_s.item())
                losses_side = np.append(losses_side, losses_side_s.item())
                losses_sp = np.append(losses_sp, losses_sp_s.item())
                losses_body = np.append(losses_body, losses_body_s.item())
                losses_detail = np.append(losses_detail, losses_detail_s.item())


                steps_list = np.append(steps_list, num_steps)

                # Reset the cumulative step loss
                losses_comb_s *= 0
                losses_side_s *= 0
                losses_sp_s *= 0
                losses_body_s *= 0
                losses_detail_s *= 0


        if local_rank == 0:
                logging.info(f"Loss at epoch #{epoch}, step #{num_steps}, update #{num_steps / numSteps}\n" + \
                      f"Combined: {round(losses_comb.mean(), 6)}    " \
                      f"Simple: {round(losses_sp.mean(), 6)}    " \
                      f"Side: {round(losses_side.mean(), 6)} "
                      f"Body: {round(losses_body.mean(), 6)}"
                      f"Detail: {round(losses_detail.mean(), 6)}")


        #eval

        is_eval = False
        # if epoch == 1:
        #     is_eval = True
        # if epoch > 0 and epoch < 500 and epoch % 20==0:
        #     is_eval = True
        # if epoch >= 500 and epoch < 700 and epoch % 10==0:
        #     is_eval = True
        if epoch >= 500 and epoch % 20==0:
            is_eval = True

        # if epoch > 0 :
        if is_eval:
            saveDir = args.checkpoint_dir
            if not os.path.isdir(saveDir):
                os.makedirs(saveDir)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(saveDir,
                            f'state_dict_epoch_{epoch}_step_{num_steps}_dice_{str(best_dice).replace(".", "_")}.pt'))
            logging.info("Saving model")


def epoch_evaluating(model, test_dataloader, device, criteria_metrics, local_rank, last_Dicescore):
    # Switch model to evaluation mode
    from monai.utils import set_determinism
    set_determinism(1)
    model.eval()
    out_pred_final = torch.FloatTensor().cuda(local_rank)
    out_pred_7 = torch.FloatTensor().cuda(local_rank)  # Tensor stores prediction values
    out_pred_14 = torch.FloatTensor().cuda(local_rank)
    out_pred_28 = torch.FloatTensor().cuda(local_rank)
    out_pred_56 = torch.FloatTensor().cuda(local_rank)
    out_gt = torch.FloatTensor().cuda(local_rank)  # Tensor stores groundtruth values
    savepath = './prediction/' + args.dataset_name
    # cal_fm = CalFM(1)  # cal是一个对象

    with torch.no_grad():  # Turn off gradient
        # For each batch
        test_output_root = os.path.join(args.job_name, savepath)

        for step, (images, masks, index) in enumerate(test_dataloader):
            # print(len(test_dataloader))

            # Move images, labels to device (GPU)
            input = images.cuda(local_rank)
            masks = masks.cuda(local_rank)
            # input = images * 2 - 1
            preds = torch.zeros((input.shape[0], args.num_ens, input.shape[2], input.shape[3])).cuda(local_rank)
            for i in range(args.num_ens):
                # loss_simple, input_side_out, body_pre, detail_pre = model.module(input,masks)
                # preds[:, i:i + 1, :, :] = input_side_out[0]
                preds[:, i:i + 1, :, :],_ = model.module.sample(input)
            preds_mean = preds.mean(dim=1)
            # out_pred_final = torch.cat((out_pred_final, preds_mean), 0)
            # preds_mean = preds_mean

            preds_mean[preds_mean < 0.3] = 0

            # preds_mean1 = preds_mean.data.cpu().numpy()
            out_pred_final = torch.cat((out_pred_final, preds_mean), 0)
            # Update groundtruth values
            out_gt = torch.cat((out_gt, masks), 0)

            if local_rank == 0 and step % args.print_freq == 0 or step == len(test_dataloader) - 1:
                logging.info(
                    "val: Step {:03d}/{:03d}".format(step, len(test_dataloader) - 1))

    _recallC, _specificityC, _precisionC, _F1C, _F2C, _ACC_overallC, _IoU_polyC, _IoU_bgC, _IoU_meanC, _MSD, _ASD = criteria_metrics(
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
    # Clear memory
    del images, masks, out_pred_final, out_gt, out_pred_7, out_pred_14, out_pred_28, out_pred_56
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # return validation loss, and metric score
    return score_metricsC

if __name__ == '__main__':
    main()
