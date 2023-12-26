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

import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from fit import fit, benchmark_fit
from loss import GeneratorLoss, DiscriminatorLoss, BenchmarkLoss
from models.unet import UNetDiscriminator
from models.fixup import FixUpUnet
from dataset import ShadowsDataset

def main(args):
    # Reproducibility setup
    gen = set_reproducible(seed=args.seed, set=args.reproducible)

    # Tensorboard setup
    writer = SummaryWriter(filename_suffix=('_' + args.experiment_tag))
    layout = {'Generator': {
                            'total loss': ['Multiline', ['g_train', 'g_val']],
                            'shadow loss': ['Multiline', ['shadow_train', 'shadow_val']],
                            'ground loss': ['Multiline', ['ground_train', 'ground_val']],
                            'adversarial loss': ['Multiline', ['gan_train', 'gan_val']],
                            'shadow rmse score': ['Multiline', ['shadow_rmse']],
                            'shadow si-rmse score': ['Multiline', ['shadow_si_rmse']],
                            'ground rmse score': ['Multiline', ['ground_rmse']],
                            'ground si-rmse score': ['Multiline', ['ground_si_rmse']],
                            },
              'Discriminator': {
                            'total loss': ['Multiline', ['d_train', 'd_val']],
                            'fake loss': ['Multiline', ['fake_train', 'fake_val']],
                            'real loss': ['Multiline', ['real_train', 'real_val']],
                            }}
    writer.add_custom_scalars(layout)

    # Dataset setup
    dataset = ShadowsDataset(args, train=True)
    train_len = int(args.train_val_split * dataset.__len__())
    val_len = dataset.__len__() - train_len
    splits_len = [train_len, val_len]
    train_split, val_split = random_split(dataset, splits_len, generator=gen)

    # Dataloader setup
    loader_train = DataLoader(train_split,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=0,
                              generator=gen)
    loader_val = DataLoader(val_split,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=0,
                            generator=gen)

    # GAN model and losses setup
    model_G = None
    model_D = None
    optimizer_D = None
    loss_D = None
    loss_benchmark = None
    model_G = FixUpUnet(in_res=args.in_res, detection_input=args.use_detection_input).to(args.device)
    optimizer_G = torch.optim.AdamW(model_G.parameters(),
                                    lr=args.learning_rate_g,
                                    betas=args.adam_betas)
    loss_G = GeneratorLoss(args)
    loss_benchmark = BenchmarkLoss(args.in_res)
    if args.gan:
        model_D = UNetDiscriminator(in_res=args.in_res).to(args.device)
        optimizer_D = torch.optim.AdamW(model_D.parameters(),
                                        lr=args.learning_rate_d,
                                        betas=args.adam_betas)
        loss_D = DiscriminatorLoss(args)

    # Resuming training
    if args.checkpoint_restart:
        print('Restarting from checkpoint at epoch ' + str(args.checkpoint_epoch))
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, args.checkpoint_tag, \
                                  args.checkpoint_tag + '_' + str(args.checkpoint_epoch) + \
                                     '.ckpt'), map_location=args.device)
        model_G.load_state_dict(checkpoint['g_model_state'])
        optimizer_G.load_state_dict(checkpoint['g_optimizer_state'])
        if args.gan:
            model_D.load_state_dict(checkpoint['d_model_state'])
            optimizer_D.load_state_dict(checkpoint['d_optimizer_state'])

    loss_log = {'train': {}, 'validation': {}}
    e_start = 0
    if args.checkpoint_restart:
        e_start = args.checkpoint_epoch

    # Starting training for E epochs
    for e in range(e_start, args.epochs):
        model_G.train()
        if args.gan: model_D.train()

        # Fitting the models to the dataset
        loss_train = fit(model_G, optimizer_G, loss_G,
                         model_D, optimizer_D, loss_D,
                         loader_train, e, writer, args, train=True)

        for key in loss_train:
            if key not in loss_log['train']:
                loss_log['train'][key] = []
            loss_log['train'][key].append(loss_train[key])

        # Validation
        with torch.no_grad():
            model_G.eval()
            if args.gan: model_D.eval()
            loss_val = fit(model_G, optimizer_G, loss_G,
                           model_D, optimizer_D, loss_D,
                           loader_val, e, writer, args, train=False)
            for key in loss_val:
                if key not in loss_log['validation']:
                    loss_log['validation'][key] = []
                loss_log['validation'][key].append(loss_val[key])

        # Tensorboard value logging
        writer.add_scalar('g_train', loss_train['total_g'], e)
        writer.add_scalar('g_val', loss_val['total_g'], e)
        writer.add_scalar('ground_train', loss_train['ground'], e)
        writer.add_scalar('ground_val', loss_val['ground'], e)
        writer.add_scalar('shadow_train', loss_train['shadow'], e)
        writer.add_scalar('shadow_val', loss_val['shadow'], e)
        writer.add_scalar('gan_train', loss_train['adversarial'], e)
        writer.add_scalar('gan_val', loss_val['adversarial'], e)
        writer.add_scalar('d_train', loss_train['total_d'], e)
        writer.add_scalar('d_val', loss_val['total_d'], e)
        writer.add_scalar('fake_train', loss_train['fake_d'], e)
        writer.add_scalar('fake_val', loss_val['fake_d'], e)
        writer.add_scalar('real_train', loss_train['real_d'], e)
        writer.add_scalar('real_val', loss_val['real_d'], e)
        writer.flush()

        # RMSE benchmark validation every 5 epochs
        if (((e+1) % 5 == 0) or (e == args.epochs - 1)):
            with torch.no_grad():
                bench = benchmark_fit(args.device, model_G, loss_benchmark, loader_val)
                writer.add_scalar('shadow_rmse', bench[0], e)
                writer.add_scalar('shadow_si_rmse', bench[1], e)
                writer.add_scalar('ground_rmse', bench[2], e)
                writer.add_scalar('ground_si_rmse', bench[3], e)

        # Checkpoint saving
        if (((e+1) % args.checkpoint_interval == 0) or (e == args.epochs - 1)):
            save_dict = {'epoch': e,
                         'g_model_state': model_G.state_dict().copy(),
                         'g_optimizer_state': optimizer_G.state_dict().copy(),
                         'losses': loss_log,
                         'in_res': args.in_res,
                         'args': args}
            if args.gan:
                save_dict['d_model_state'] = model_D.state_dict().copy()
                save_dict['d_optimizer_state'] = optimizer_D.state_dict().copy()
            checkpoint_path = os.path.join(args.checkpoint_dir, args.experiment_tag)
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            checkpoint_name = args.experiment_tag + '_' + str(e + 1) + '.ckpt'
            filename = os.path.join(checkpoint_path, checkpoint_name)
            torch.save(save_dict, filename)
            print('Saving Checkpoint')
    writer.close()


# Reproducibility for Python, NumPy, and PyTorch
def set_reproducible(seed=torch.initial_seed(), set=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = set
        torch.backends.cudnn.benchmark = not set
    return generator


# Training flags and arguments
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_tag', type=str)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--reproducible', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint_interval', type=int, default=20)
    parser.add_argument('--in_res', type=int, default=128)
    parser.add_argument('--checkpoint_restart', action='store_true', default=False)
    parser.add_argument('--checkpoint_tag', type=str, default='debug')
    parser.add_argument('--checkpoint_epoch', type=int, default=20)
#-------------------------------
    parser.add_argument('--use_synthetic', action='store_true', default=False)
    parser.add_argument('--use_detection', action='store_true', default=False)
    parser.add_argument('--use_srd', action='store_true', default=False)
    parser.add_argument('--use_all_augmentations', action='store_true', default=False)
    parser.add_argument('--use_ess_augmentation', action='store_true', default=False)
    parser.add_argument('--use_nc_augmentation', action='store_true', default=False)
    parser.add_argument('--use_ni_augmentation', action='store_true', default=False)
    parser.add_argument('--use_direct_prediction', action='store_true', default=False)
    parser.add_argument('--use_detection_input', action='store_true', default=False)
    parser.add_argument('--use_ground_loss', action='store_true', default=False)
#-------------------------------
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--city_dataset_dir', type=str)
    parser.add_argument('--istd_dataset_dir', type=str)
    parser.add_argument('--srd_dataset_dir', type=str)
    parser.add_argument('--desoba_dataset_dir', type=str)
    parser.add_argument('--sbu_dataset_dir', type=str)
    parser.add_argument('--debug_dataset', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--train_val_split', type=float, default=0.9)
    parser.add_argument('--learning_rate_g', type=float, default=0.0001)
    parser.add_argument('--learning_rate_d', type=float, default=0.0001)
    parser.add_argument('--adam_betas', type=float, nargs=2, default=[0.9, 0.999])
    parser.add_argument('--gan', action='store_true', default=False)
    parser.add_argument('--shadow_loss_weight', type=float, default=0)
    parser.add_argument('--ground_loss_weight', type=float, default=0)
    parser.add_argument('--gan_loss_weight', type=float, default=0, help='for g')
    args = parser.parse_args()

    # Log and start
    print(args)
    main(args)
