import torch
from torch import nn
import torch.nn.functional as F

# Simple loss function (L_simple) (MSE Loss)
# Inputs:
#   epsilon - True epsilon values of shape (N, C, L, W)
#   epsilon_pred - Predicted epsilon values of shape (N, C, L, W)
# Outputs:
#   Vector loss value for each item in the entire batch
def loss_simple(epsilon, epsilon_pred):
    return ((epsilon_pred - epsilon) ** 2).flatten(1, -1).mean(-1)


# Variational Lower Bound loss function which computes the
# KL divergence between two gaussians
# Formula derived from: https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
# Inputs:
#   t - Values of t of shape (N)
#   mean_real - The mean of the real distribution of shape (N, C, L, W)
#   mean_fake - Mean of the predicted distribution of shape (N, C, L, W)
#   var_real - Variance of the real distribution of shape (N, C, L, W)
#   var_fake - Variance of the predicted distribution of shape (N, C, L, W)
# Outputs:
#   Loss vector for each part of the entire batch
def loss_vlb_gauss(t, mean_real, mean_fake, var_real, var_fake):
    std_real = torch.sqrt(var_real)
    std_fake = torch.sqrt(var_fake)

    # Note:
    # p (mean_real, std_real) - Distribution we want the model to predict
    # q (mean_fake, std_fake) - Distribution the model is predicting
    output = (torch.log(std_fake / std_real) \
              + ((var_real) + (mean_real - mean_fake) ** 2) / (2 * (var_fake)) \
              - torch.tensor(1 / 2)) \
        .flatten(1, -1).mean(-1)

    return output


# Combined loss
# Inputs:
#   epsilon - True epsilon values of shape (N, C, L, W)
#   epsilon_pred - Predicted epsilon values of shape (N, C, L, W)
#   x_t - The noised image at time t of shape (N, C, L, W)
#   t - The value timestep of shape (N)
# Outputs:
#   Loss as a scalar over the entire batch
def lossFunct(model, epsilon, epsilon_pred, v, x_0, x_t, t, Lambda):
    # Put the data on the correct device
    x_0 = x_0.to(epsilon_pred.device)
    x_t = x_t.to(epsilon_pred.device)

    """
    There's one important note I looked` passed when reading the original
    Denoising Diffusion Probabilistic Models paper. I noticed that L_simple
    was high on low values of t but low on high values of t. I thought
    this was an issue, but it is not. As stated in the paper

    "In particular, our diffusion process setup in Section 4 causes the 
    simplified objective to down-weight loss terms corresponding to small t.  
    These terms train the network to denoise data with very small amounts of 
    noise, so it is beneficial to down-weight them so that the network can 
    focus on more difficult denoising tasks at larger t terms"
    (page 5 part 3.4)
    """

    # Get the mean and variance from the model

    mean_t_pred = model.module.noise_to_mean(epsilon_pred, x_t, t, True)
    var_t_pred = model.module.vs_to_variance(v, t)

    ### Preparing for the real normal distribution

    # Get the scheduler information

    beta_t = model.module.scheduler.sample_beta_t(t)
    a_bar_t = model.module.scheduler.sample_a_bar_t(t)
    a_bar_t1 = model.module.scheduler.sample_a_bar_t1(t)
    beta_tilde_t = model.module.scheduler.sample_beta_tilde_t(t)
    sqrt_a_bar_t1 = model.module.scheduler.sample_sqrt_a_bar_t1(t)
    sqrt_a_t = model.module.scheduler.sample_sqrt_a_t(t)

    # Get the true mean distribution
    mean_t = ((sqrt_a_bar_t1 * beta_t) / (1 - a_bar_t)) * x_0 + \
             ((sqrt_a_t * (1 - a_bar_t1)) / (1 - a_bar_t)) * x_t

    # Get the losses
    loss_sp = loss_simple(epsilon, epsilon_pred)
    # pre = epsilon_pred - epsilon
    # salloss = comput_loss(pre,x_0)
    loss_vlb = loss_vlb_gauss(t, mean_t, mean_t_pred.detach(), beta_tilde_t, var_t_pred) * Lambda

    # Get the combined loss
    loss_comb = loss_sp + loss_vlb

    # # Update the loss storage for importance sampling
    # if self.use_importance:
    #     with torch.no_grad():
    #         t = t.detach().cpu().numpy()
    #         loss = loss_vlb.detach().cpu()
    #         self.update_losses(loss, t)
    #
    #         # Have 10 loss values been sampled for each value of t?
    #         if np.sum(self.losses_ct) == self.losses.size - 20:
    #             # The losses are based on the probability for each
    #             # value of t
    #             p_t = np.sqrt((self.losses ** 2).mean(-1))
    #             p_t = p_t / p_t.sum()
    #             loss = loss / torch.tensor(p_t[t], device=loss.device)
    #         # Otherwise, don't change the loss values

    # Return the losses
    return loss_comb.mean(), loss_sp.mean(), loss_vlb.mean()

def dice_loss(pre, target, smooth = 1.):
    pre = F.sigmoid(pre)
    pre = pre.view(-1)
    target = target.view(-1)
    intersection = (pre * target).sum()
    dice = (2. * intersection + smooth)/(pre.sum() + target.sum() + smooth)
    return 1-dice
def structure_loss(pred, mask, type = None):
    mask = mask.to(pred.device)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    n, c, _, _ = pred.shape
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    if type == 'bce+iou':
    # allloss = wiou.mean() + wbce.mean()
        allloss = wbce.mean() + wiou.mean()
        return allloss
    elif  type == 'bce':
    # allloss = wiou.mean() + wbce.mean()
        allloss = wbce.mean()
        return allloss
    elif type == 'bce+iou+dice':
        dice = dice_loss(pred, mask)
        allloss = wbce.mean() + wiou.mean() + dice.mean()
        return allloss
    else:
        raise Exception
# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def adaptive_pixel_intensity_loss(pred, mask):
    w1 = torch.abs(F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1) - mask)
    w2 = torch.abs(F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7) - mask)
    w3 = torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    omega = 1 + 0.5 * (w1 + w2 + w3) * mask

    bce = F.binary_cross_entropy_with_logits(pred, mask, reduce=None)
    abce = (omega * bce).sum(dim=(2, 3)) / (omega + 0.5).sum(dim=(2, 3))

    inter = ((pred * mask) * omega).sum(dim=(2, 3))
    union = ((pred + mask) * omega).sum(dim=(2, 3))
    aiou = 1 - (inter + 1) / (union - inter + 1)

    mae = F.l1_loss(pred, mask, reduce=None)
    amae = (omega * mae).sum(dim=(2, 3)) / (omega - 1 + 1e-5).sum(dim=(2, 3))

    # return (0.7 * abce + 0.7 * aiou + 0.7 * amae).mean()
    return (0.7 * abce + 0.7 * aiou).mean()


def comput_loss(side_out, target, type):
    if isinstance(side_out,list):
        sal_loss1 = structure_loss(side_out[0], type)
        sal_loss2 = structure_loss(side_out[1], type)
        sal_loss3 = structure_loss(side_out[2], type)
        sal_loss4 = structure_loss(side_out[3], type)
        side_out_loss = sal_loss1 / 2 + sal_loss2 / 4 + sal_loss3 / 8 + sal_loss4 / 8
        # salloss = sal_loss1
        return side_out_loss
    else:
        sal_loss1 = structure_loss(side_out, target, type)
        return sal_loss1

