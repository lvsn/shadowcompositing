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

from tqdm import tqdm
import torchvision

# Train or test the model for one whole epoch
def fit(model_G, optimizer_G, loss_fn_G, model_D, optimizer_D, loss_fn_D,
        loader, epoch, writer, args, train=True):

    # TQDM display and Tensorboard
    running_losses = {'total_g': 0,
                      'shadow': 0,
                      'ground': 0,
                      'adversarial': 0,
                      'total_d': 0,
                      'fake_d': 0,
                      'real_d': 0}
    total_g = 0
    total_d = 0 if args.gan else 'N/A'

    # TQDM display
    color = 'green' if train else 'red'
    loop = tqdm(loader, total=len(loader), leave=True, ncols=120, colour=color,
                desc=f'Epoch [{epoch + 1}/{args.epochs}]')

    for idx, data_sample in enumerate(loop):
        # Selecting only necessary dataset layers
        sample = {}
        aug_keys = ['net_input', 'net_target', 'shadow_target']
        for key in aug_keys:
            sample[key] = data_sample[key].to(args.device)

        # Generator inference
        pred_G = model_G(2 * sample['net_input'] - 1)

        # Baseline that predicts the background composite outright
        if args.use_direct_prediction:
            pred_G_D = pred_G[:, :3]
        # Gain map prediction (main baseline)
        else:
            pred_G_D = sample['net_input'][:, :3] * pred_G[:, :3]

        # Discriminator inference
        pred_D = model_D(2 * pred_G_D - 1) if args.gan else None
        loss_G = loss_fn_G(pred_G, sample, pred_D=pred_D)

        # Generator backpropagation
        if train:
            optimizer_G.zero_grad()
            loss_G['total'].backward()
            optimizer_G.step()
            loss_G['total'].detach()

        # Tensorboard live compositing for debugging
        elif idx == max(len(loader) - 2, 0):
            res = args.in_res
            batch = sample['net_input'].shape[0]
            mask = sample['net_input'][:, 3]
            mask_3c = mask.view((batch, 1, res, res)).repeat((1, 3, 1, 1))
            rgb_input = sample['net_input'][:, 0:3]
            if args.use_direct_prediction:
                rgb_pred = pred_G[:, 0:3]
            else:
                ground_pred = rgb_input * pred_G[:, 0:3]
                rgb_pred = ground_pred * mask_3c + rgb_input * (1 - mask_3c)
            rgb_target = sample['net_target']
            shadow_pred = (pred_G[:, 3] * mask).view((batch, 1, res, res)).repeat((1, 3, 1, 1))
            shadow_target = (sample['shadow_target'] * mask).view((batch, 1, res, res)).repeat((1, 3, 1, 1))
            shadow_mask = sample['net_input'][:, 3].view((batch, 1, res, res)).repeat((1, 3, 1, 1))

            rgb_in_grid = torchvision.utils.make_grid((rgb_input ** (1/2.2)).clamp(0, 1))
            rgb_pred_grid = torchvision.utils.make_grid((rgb_pred ** (1/2.2)).clamp(0, 1))
            rgb_gt_grid = torchvision.utils.make_grid((rgb_target ** (1/2.2)).clamp(0, 1))
            shadow_pred_grid = torchvision.utils.make_grid(shadow_pred)
            shadow_gt_grid = torchvision.utils.make_grid(shadow_target)
            shadow_mask_grid = torchvision.utils.make_grid(shadow_mask)

            writer.add_image('RGB Input', rgb_in_grid, global_step=(epoch + 1))
            writer.add_image('RGB Pred', rgb_pred_grid, global_step=(epoch + 1))
            writer.add_image('RGB Target', rgb_gt_grid, global_step=(epoch + 1))
            writer.add_image('Shadow Pred', shadow_pred_grid, global_step=(epoch + 1))
            writer.add_image('Shadow Target', shadow_gt_grid, global_step=(epoch + 1))
            writer.add_image('To Insert', shadow_mask_grid, global_step=(epoch + 1))
            writer.flush()

        # Discriminator inference and backpropagation
        if args.gan:
            pred_G_D = model_G(2 * sample['net_input'] - 1)[:, :3]

            # Baseline that predicts the background composite outright
            if args.use_full_comp_d:
                res = args.in_res
                batch = sample['net_input'].shape[0]
                mask = sample['net_input'][:, 3]
                obj_shadow = mask.view((batch, 1, res, res)).repeat((1, 3, 1, 1))
                gain = pred_G_D
                bg = sample['net_input'][:, :3]
                pred_G_D = (obj_shadow * bg * gain + (1 - obj_shadow) * bg)
                pred_fake = model_D(2 * pred_G_D - 1)
                pred_real = model_D(2 * sample['net_target'] - 1)

            # Other baselines
            else:
                if args.use_direct_prediction:
                    pred_G_D = pred_G_D
                else:
                    pred_G_D = pred_G_D * sample['net_input'][:, :3]
                pred_fake = model_D(2 * pred_G_D - 1)
                pred_real = model_D(2 * sample['net_target'] - 1)

            # Loss calculation for the discriminator
            loss_D = loss_fn_D(pred_real, pred_fake)

            # Discriminator backpropagation
            if train:
                optimizer_D.zero_grad()
                loss_D['total'].backward()
                optimizer_D.step()

        # Loss accumulation and display
        running_losses['total_g'] += loss_G['total'].detach().item()
        total_g = round(running_losses['total_g'] / (idx + 1), 4)
        running_losses['shadow'] += loss_G['shadow'].detach().item()
        if args.use_ground_loss or args.use_direct_prediction:
            running_losses['ground'] += loss_G['ground'].detach().item()
        if args.gan:
            running_losses['adversarial'] += loss_G['adversarial'].detach().item()
            running_losses['fake_d'] += loss_D['fake'].detach().item()
            running_losses['real_d'] += loss_D['real'].detach().item()
            running_losses['total_d'] += loss_D['total'].detach().item()
            total_d = round(running_losses['total_d'] / (idx + 1), 4)
        loop.set_postfix(loss_G=total_g, loss_D=total_d)

    # Scaling of accumulated losses
    for key in running_losses:
        running_losses[key] /= len(loader)
    return running_losses


# Current RMSE on the validation set every 5 epochs
def benchmark_fit(device, model, loss, loader, get_full=False):
    loop = tqdm(loader, total=len(loader), leave=True, ncols=120,
                colour='blue', desc='Benchmarking')

    for data_sample in loop:
        # Selecting only necessary dataset layers
        sample = {}
        aug_keys = ['net_input', 'net_target', 'shadow_target']
        for key in aug_keys:
            sample[key] = data_sample[key].to(device)

        # Inference and loss accumulation
        pred = model(2 * sample['net_input'] - 1)
        loss.update(pred, sample)

    # RMSE calculation
    shadow_rmse, shadow_si_rmse, ground_rmse, ground_si_rmse = loss(get_full)

    # Average errors for the batch
    if not get_full:
        print('SHADOW DETECTION RMSE: ' + str(round(shadow_rmse, 4)) + '  |  ' +\
              'si-RMSE: ' + str(round(shadow_si_rmse, 4)))
        print('RGB GROUND RMSE: ' + str(round(ground_rmse, 4)) + '  |  ' +\
              'si-RMSE: ' + str(round(ground_si_rmse, 4)))

    return shadow_rmse, shadow_si_rmse, ground_rmse, ground_si_rmse
