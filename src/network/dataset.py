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

import numpy as np
import torch
import cv2 as cv
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as T
from datasets.city_dataset import CityShadowsDataset
from datasets.real_dataset import RealDataset

# Class that accumulates all training data and performs live augmentations
class ShadowsDataset(Dataset):
    def __init__(self, args, train=True):
        self.args = args
        self.samples = []

        # Loading shadow removal data
        istd = RealDataset(args.istd_dataset_dir, args.in_res,
                           debug=args.debug_dataset, train=train,
                           instances=False, name='ISTD')
        desoba = RealDataset(args.desoba_dataset_dir, args.in_res,
                             debug=args.debug_dataset, train=train,
                             instances=True, name='DESOBA')
        # Loading synthetic data
        if args.use_synthetic:
            blender = CityShadowsDataset(args.city_dataset_dir, args.in_res,
                                         train=train, debug=args.debug_dataset)
        # Loading shadow removal data with questionable ground truth (still beneficial, see supplemental)
        if args.use_srd:
            srd = RealDataset(args.srd_dataset_dir, args.in_res,
                              debug=args.debug_dataset, train=train,
                              instances=False, name='SRD')
        # Loading shadow detection data
        if args.use_detection:
            sbu = RealDataset(args.sbu_dataset_dir, args.in_res,
                              debug=args.debug_dataset, train=train,
                              instances=False, name='SBU')
            # ucf = RealDataset(args.ucf_dataset_dir, args.in_res,
            #                   debug=args.debug_dataset, train=train,
            #                   instances=False, name='UCF')

        # Create dictionaries for live augmentation and sampling
        datasets = { 'istd': istd,
                     'desoba': desoba }
        if args.use_synthetic:
            datasets['blender'] = blender
        if args.use_srd:
            datasets['srd'] = srd
        if args.use_detection:
            datasets['sbu'] = sbu
            # datasets['ucf'] = ucf
        encoding = { 'blender': 0,
                     'istd': 1,
                     'srd': 2,
                     'desoba': 3,
                     'sbu': 4,
                     'ucf': 5 }

        # Prepare arrays
        for name in datasets:
            for idx in range(datasets[name].__len__()):
                self.samples.append(datasets[name].__getitem__(idx))
                self.samples[-1]['source'] = encoding[name]
        self.len = len(self.samples)

    def __len__(self):
        return self.len


    # Live augmentation to create alpha-blended shadow subsets from ellipsoid primitives
    def get_blend_mask(self, mask, max_circles=15, blur=True):
        res = self.args.in_res
        filter = np.zeros((res, res))
        antifilter = np.zeros((res, res))

        # Create two random binary masks of overlapping ellipsoids
        for f in [filter, antifilter]:
            circles = int(np.random.rand() * max_circles)
            if circles > 0:
                angles = np.random.rand(circles) * 360
                axes_x = (np.random.rand(circles) * (res // 2)).astype(int)
                axes_y = (np.random.rand(circles) * (res // 2)).astype(int)
                centers_x = (np.random.rand(circles) * (res - 1)).astype(int)
                centers_y = (np.random.rand(circles) * (res - 1)).astype(int)
            for idx in range(circles):
                f = cv.ellipse(f, (centers_x[idx], centers_y[idx]),
                                  (axes_x[idx], axes_y[idx]),
                                  angles[idx], 0, 360, 1, -1)

        # Calculate the intersection for increased randomization and mask to the region to harmonize
        filter = filter * antifilter * mask

        # Simulate soft boundaries to make it more realistic
        if blur:
            filter = cv.GaussianBlur(filter, (5, 5), 1)

        # Return resulting mask in grayscale and RGB
        filter3c = filter.reshape((res, res, 1))
        filter3c = np.repeat(filter3c, 3, axis=2).astype(float)
        return filter, filter3c


    # Live data augmentation
    # Note: can be changed as desired, different proportions yield different performances
    def get_sample(self, idx):
        input_rgb = self.samples[idx]['input'].copy()
        gt_rgb = self.samples[idx]['gt'].copy()

        # ----------------------------------------------------------------------

        # blender needs no augmentation
        if self.samples[idx]['source'] == 0:
            to_insert = self.samples[idx]['mask'].copy()
            input_shadows = self.samples[idx]['input_shadows'].copy()

        # ----------------------------------------------------------------------

        elif self.samples[idx]['source'] in [4, 5]:
            input_shadows = self.samples[idx]['mask'].copy()

            if (np.random.rand() < 0.25) and (self.args.use_all_augmentations or self.args.use_ni_augmentation or self.args.use_nc_augmentation):
                # insert nothing, full shadow already there
                if ((np.random.rand() < 0.5) and (self.args.use_all_augmentations or self.args.use_ni_augmentation)) or \
                   ((not self.args.use_all_augmentations) and (not self.args.use_nc_augmentation)):
                    to_insert = np.zeros(input_shadows.shape, input_shadows.dtype)

                # try to insert full shadow that's already there
                else:
                    to_insert = input_shadows.copy()

            # try to blend shadows already there
            else:
                # subset mask
                if (np.random.rand() < 0.25) and (self.args.use_all_augmentations or self.args.use_ess_augmentation): # 22.5% subset blend
                    k = np.ones((5, 5), np.uint8)
                    bm = cv.erode(input_shadows.astype(np.uint8), k, iterations=1)
                    bm = cv.GaussianBlur(bm, (5, 5), 1)

                    # inverse subset
                    if np.random.rand() < 0.5:
                        bm = (1 - bm) * input_shadows

                # random mask
                else:
                    bm, _ = self.get_blend_mask(input_shadows.astype(np.uint8))

                to_insert = bm.astype(float)

        # ----------------------------------------------------------------------

        elif self.samples[idx]['source'] in [1, 2]:
            to_insert = self.samples[idx]['mask'].copy()

            if (np.random.rand() < 0.15) and (self.args.use_all_augmentations or self.args.use_ni_augmentation or self.args.use_nc_augmentation):
                # insert nothing, full shadow already there
                if ((np.random.rand() < 0.33) and (self.args.use_all_augmentations or self.args.use_ni_augmentation)) or \
                   ((not self.args.use_all_augmentations) and (not self.args.use_nc_augmentation)):
                    input_shadows = self.samples[idx]['mask'].copy()
                    input_rgb = gt_rgb.copy()
                    to_insert = np.zeros(to_insert.shape, to_insert.dtype)
                else:
                    # insert full shadow
                    if (np.random.rand() < 0.5):
                        input_shadows = np.zeros(to_insert.shape, to_insert.dtype)

                    # try to insert full shadow that's already there
                    else:
                        input_shadows = self.samples[idx]['mask'].copy()
                        input_rgb = gt_rgb.copy()

            # blend shadows
            else:
                # subset
                if (np.random.rand() < 0.25) and (self.args.use_all_augmentations or self.args.use_ess_augmentation): # 16% of submask blending
                    k = np.ones((5, 5), np.uint8)
                    bm = cv.erode(to_insert.astype(np.uint8), k, iterations=1)
                    bm = cv.GaussianBlur(bm, (5, 5), 1)

                    # inverse subset
                    if np.random.rand() < 0.5:
                        bm = (1 - bm) * to_insert

                    bm_3c = bm.reshape((bm.shape[0], bm.shape[1], 1))
                    bm_3c = np.repeat(bm_3c, 3, axis=2).astype(float)

                # random blending
                else:
                    bm, bm_3c = self.get_blend_mask(to_insert.astype(np.uint8))

                input_rgb[bm > 0] = bm_3c[bm > 0] * gt_rgb[bm > 0] + \
                                    (1 - bm_3c[bm > 0]) * input_rgb[bm > 0]
                input_shadows = bm.astype(float)

        # ----------------------------------------------------------------------

        elif self.samples[idx]['source'] == 3:
            if (self.args.use_all_augmentations or self.args.use_nc_augmentation or self.args.use_ni_augmentation):
                # select how many instances to use
                n_instances = max(int(np.random.rand() * len(self.samples[idx]['instances'])), 1)
                instances = np.random.choice(np.arange(len(self.samples[idx]['instances'])), n_instances)
            else:
                n_instances = len(self.samples[idx]['instances'])
                instances = np.arange(n_instances)

            to_insert = self.samples[idx]['mask'].copy()
            input_shadows = np.zeros(to_insert.shape, to_insert.dtype)

            # randomize instances individually
            for idx, instance in enumerate(self.samples[idx]['instances']):
                # if instance not chosen to blend
                if idx not in instances:
                    # insert nothing, full shadow already there
                    if (np.random.rand() < 0.33) or \
                       ((not self.args.use_all_augmentations) and (not self.args.use_nc_augmentation)):
                        input_rgb[instance > 0] = gt_rgb[instance > 0]
                        input_shadows[instance > 0] = 1
                        to_insert[instance > 0] = 0
                        continue
                    # insert full shadow
                    elif (np.random.rand() < 0.5):
                        continue
                    # try to insert full shadow that's already there
                    else:
                        input_rgb[instance > 0] = gt_rgb[instance > 0]
                        input_shadows[instance > 0] = 1
                        continue

                # if instance was selected...
                # insert subset
                if (np.random.rand() < 0.25) and (self.args.use_all_augmentations or self.args.use_ess_augmentation): # 18% subset blending (9% each)
                    k = np.ones((5, 5), np.uint8)
                    bm = cv.erode(instance.astype(np.uint8), k, iterations=1)
                    bm = cv.GaussianBlur(bm, (5, 5), 1)

                    if np.random.rand() < 0.5:
                        bm = (1 - bm) * instance

                    bm_3c = bm.reshape((bm.shape[0], bm.shape[1], 1))
                    bm_3c = np.repeat(bm_3c, 3, axis=2).astype(float)

                # random blend
                else:
                    bm, bm_3c = self.get_blend_mask(instance.astype(np.uint8))

                input_rgb[bm > 0] = bm_3c[bm > 0] * gt_rgb[bm > 0] + \
                                    (1 - bm_3c[bm > 0]) * input_rgb[bm > 0]
                input_shadows = np.maximum(input_shadows, bm)

        return input_rgb, to_insert, gt_rgb, input_shadows


    # Data loading
    def __getitem__(self, idx):
        sample = {}

        # Live augmentation
        input_rgb, to_insert, gt_rgb, input_shadows = self.get_sample(idx)

        # Reshaping
        input_rgb = torch.tensor(np.moveaxis(input_rgb, -1, 0)).float()
        to_insert = torch.tensor(to_insert.reshape((1, to_insert.shape[0], to_insert.shape[1]))).float()
        gt_rgb = torch.tensor(np.moveaxis(gt_rgb, -1, 0)).float()
        input_shadows = torch.tensor(input_shadows).float()
        input_rgb = torch.cat((input_rgb, to_insert), dim=0)

        # Dictionary for the fitting loop
        sample = { 'net_input': input_rgb,
                   'net_target': gt_rgb,
                   'shadow_target': input_shadows }

        # Image flipping augmentation
        # 50% of flipping horizontally
        if np.random.rand() < 0.5:
            for key in ['net_input', 'net_target', 'shadow_target']:
                sample[key] = T.hflip(sample[key].clone())

        # Input for the baseline that uses our network's gain with MTMT's detection
        if self.args.use_detection_input:
            shadows = sample['shadow_target'].clone().view((1, sample['shadow_target'].shape[0], sample['shadow_target'].shape[1]))
            sample['net_input'] = torch.cat((sample['net_input'], shadows), dim=0)

        return sample
