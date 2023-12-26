"""
Shadow Harmonization for Realisitc Compositing (c)
by Lucas Valença, Jinsong Zhang, Michaël Gharbi,
Yannick Hold-Geoffroy and Jean-François Lalonde.

Developed at Université Laval in collaboration with Adobe, for more
details please see <https://lvsn.github.io/shadowcompositing/>.

Work published at ACM SIGGRAPH Asia 2023. Full open access at the ACM
Digital Library, see <https://dl.acm.org/doi/10.1145/3610548.3618227>.

This code is licensed under a Creative Commons
Attribution-NonCommercial 4.0 International License.

You should have received a copy of the license along with this
work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.
"""

import torch
import torch.nn as nn
import numpy as np

# Losses applied to the generator network
class GeneratorLoss(nn.Module):
    def __init__(self, args):
        super(GeneratorLoss, self).__init__()
        self.args = args

        # Shadow detection loss
        self.shadow_w = args.shadow_loss_weight
        self.shadow = nn.L1Loss()

        # Harmonization loss
        if args.use_ground_loss or args.use_direct_prediction:
            self.ground_w = args.ground_loss_weight
            self.ground = nn.L1Loss()
        else:
            self.ground_w = 0

        # Adversarial loss
        if args.gan:
            self.gan_w = args.gan_loss_weight
            self.gan = nn.MSELoss()
        else:
            self.gan_w = 0

    # Usage of the generator losses
    def __call__(self, pred_G, target, pred_D=None):
        res = self.args.in_res
        batch = target['net_input'].shape[0]

        # Masks of the region to harmonize in grayscale and RGB for vectorization
        mask = target['net_input'][:, 3]
        mask_3c = mask.view((batch, 1, res, res)).repeat((1, 3, 1, 1))

        # Inputs and targets for harmonization
        rgb_input = target['net_input'][:, :3]
        rgb_target = target['net_target']

        # Inputs and targets for shadow detection
        shadow_pred = pred_G[:, 3] * mask
        shadow_target = target['shadow_target'] * mask
        shadow_loss = self.shadow(shadow_pred, shadow_target)

        if self.args.use_ground_loss:
            # Baseline that predicts the background composite directly instead of using a gain map
            if self.args.use_direct_prediction:
                ground_pred = pred_G[:, 0:3]
                ground_target = rgb_target
            # Gain map prediction (main baseline)
            else:
                ground_pred = rgb_input * pred_G[:, 0:3] * mask_3c
                ground_target = rgb_target * mask_3c
            ground_loss = self.ground(ground_pred, ground_target)
        else:
            ground_loss = 0

        # Adversarial loss for harmonization
        if self.args.gan:
            gan_target = torch.ones(pred_D.shape).to(self.args.device)
            gan_loss = self.gan(pred_D, gan_target)
        else:
            gan_loss = 0

        # Summation of losses (keeping it close to 1 at first)
        total_loss = self.ground_w * ground_loss + \
                     self.shadow_w * shadow_loss + \
                     self.gan_w * gan_loss

        # Output dictionary
        loss_dict = {'total': total_loss,
                     'shadow': shadow_loss,
                     'ground': ground_loss,
                     'adversarial': gan_loss}
        return loss_dict


# Losses applied to the discriminator network for harmonization
class DiscriminatorLoss(nn.Module):
    def __init__(self, args):
        super(DiscriminatorLoss, self).__init__()
        self.args = args
        self.loss = nn.MSELoss()

    def __call__(self, pred_real, pred_fake):
        real_target = torch.ones(pred_real.shape).to(self.args.device)
        fake_target = torch.zeros(pred_fake.shape).to(self.args.device)

        real_loss = self.loss(pred_real, real_target)
        fake_loss = self.loss(pred_fake, fake_target)
        total_loss = 0.5 * (real_loss + fake_loss)

        loss_dict = {'total': total_loss,
                     'fake': fake_loss,
                     'real': real_loss}

        return loss_dict


# Loss used to calculate the current RMSE on the validation set every 5 epochs
class BenchmarkLoss(nn.Module):
    def __init__(self, in_res):
        super(BenchmarkLoss, self).__init__()
        self.rmse = RMSELoss()
        self.shadow_rmse = []
        self.shadow_si_rmse = []
        self.ground_rmse = []
        self.ground_si_rmse = []
        self.in_res = in_res

    # Accumulation of error per batch
    def update(self, pred, target):
        res = self.in_res
        batch = target['net_input'].shape[0]

        # Masks of the region to harmonize in grayscale and RGB for vectorization
        mask = target['net_input'][:, 3]
        mask_3c = mask.view((batch, 1, res, res)).repeat((1, 3, 1, 1))

        # Inputs and targets for harmonization
        rgb_input = target['net_input'][:, 0:3]
        rgb_target = target['net_target']
        ground_pred = rgb_input * pred[:, 0:3] * mask_3c
        ground_target = rgb_target * mask_3c

        # Inputs and targets for shadow detection
        shadow_pred = pred[:, 3] * mask
        shadow_target = target['shadow_target'] * mask

        for idx in range(batch):
            # Shadow detection error
            shadow_rmse = self.rmse(shadow_pred[idx], shadow_target[idx])
            shadow_si_rmse = self.rmse(shadow_pred[idx], shadow_target[idx], si=True)

            # Harmonization error
            ground_rmse = self.rmse(ground_pred[idx], ground_target[idx])
            ground_si_rmse = self.rmse(ground_pred[idx], ground_target[idx], si=True)

            # Store values per sample
            self.shadow_rmse.append(shadow_rmse.item())
            self.shadow_si_rmse.append(shadow_si_rmse.item())
            self.ground_rmse.append(ground_rmse.item())
            self.ground_si_rmse.append(ground_si_rmse.item())


    def __call__(self, get_full=False):
        # Returns just the average of the batch
        if not get_full:
            shadow_rmse = np.array(self.shadow_rmse).mean()
            shadow_si_rmse = np.array(self.shadow_si_rmse).mean()
            ground_rmse = np.array(self.ground_rmse).mean()
            ground_si_rmse = np.array(self.ground_si_rmse).mean()

        # Returns per-sample results
        else:
            shadow_rmse = self.shadow_rmse.copy()
            shadow_si_rmse = self.shadow_si_rmse.copy()
            ground_rmse = self.ground_rmse.copy()
            ground_si_rmse = self.ground_si_rmse.copy()

        # Reset for next batch
        self.shadow_rmse = []
        self.shadow_si_rmse = []
        self.ground_rmse = []
        self.ground_si_rmse = []

        return shadow_rmse, shadow_si_rmse, ground_rmse, ground_si_rmse


# Root mean squared error metrics used in the benchmarking above
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def __call__(self, src, target, si=False):
        # Scale-invariance option
        if si:
            src = src.flatten(start_dim=1)
            target = target.flatten(start_dim=1)

            for idx, _ in enumerate(src):
                alpha = torch.dot(target[idx], src[idx])
                alpha /= torch.dot(src[idx], src[idx])
                src[idx] *= alpha

        return torch.sqrt(self.loss(src, target) + np.finfo(np.float32).eps)
