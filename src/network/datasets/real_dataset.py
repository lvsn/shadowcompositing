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
import numpy as np
from tqdm import tqdm
import imageio as iio
from torch.utils.data.dataset import Dataset
import cv2

# Shadow removal and detection datasets to augment live
# Used by the Dataset class
class RealDataset(Dataset):
    def __init__(self, root_dir, in_res, train=True, debug=False, instances=False, name=''):
        self.samples = []

        # Preparing filenames
        root = root_dir
        subset = 'train' if train else 'test'
        filenames = os.listdir(os.path.join(root, subset, subset + '_A'))
        if debug:
            filenames = filenames[:50]
        self.len = len(filenames)
        self.res = in_res
        self.name = name

        # Loading samples
        for f in tqdm(filenames, total=self.len, leave=True, colour='blue',
                      desc='Loading '+ name + ': ', ncols=90):
            paths = { 'mask': os.path.join(root, subset, subset + '_B', f),
                      'input': os.path.join(root, subset, subset + '_C', f),
                      'gt': os.path.join(root, subset, subset + '_A', f) }

            # Detection-only datasets (no removal data)
            if self.name in ['SBU', 'UCF']:
                paths['mask'] = paths['mask'].replace('.jpg', '.png')

            # Datasets with individual shadow instance annotations
            if instances:
                instance_dir = f.replace('.png', '')
                paths['instance'] = os.path.join(root, subset, subset + '_B', instance_dir)
            self.samples.append(self.prepare_sample(paths))

    # Processing samples
    def prepare_sample(self, paths):
        # Background image (pre-augmentation)
        rgb_input = iio.v2.imread(paths['input'])
        if rgb_input.shape[2] == 4:
            rgb_input = rgb_input[:, :, :3].reshape((rgb_input.shape[0], rgb_input.shape[1], 3))

        # Harmonized target ground truth
        rgb_target = iio.v2.imread(paths['gt'])
        if rgb_target.shape[2] == 4:
            rgb_target = rgb_target[:, :, :3].reshape((rgb_target.shape[0], rgb_target.shape[1], 3))

        # Ground truth shadows in the background image (shadow detection target)
        # Note: per-usage input image and mask will be obtained with live augmentation
        mask = iio.v2.imread(paths['mask'])
        if mask.ndim == 3:
            mask = np.mean(mask, axis=2)

        # Resizing and tone-mapping for the network's intended distribution
        rgb_input = cv2.resize(rgb_input, (self.res, self.res))
        rgb_target = cv2.resize(rgb_target, (self.res, self.res))
        mask = cv2.resize(mask, (self.res, self.res))
        out_sample = { 'input': (rgb_input.astype(float) / 255.) ** 2.2,
                       'gt': (rgb_target.astype(float) / 255.) ** 2.2,
                       'mask': mask.astype(float) / 255. }

        # Handling individual shadow instances
        if 'instance' in paths:
            instance_masks = []
            instance_filenames = os.listdir(paths['instance'])
            for instance_name in instance_filenames:
                instance_path = os.path.join(paths['instance'], instance_name)
                instance = iio.v2.imread(instance_path)
                if instance.ndim == 3:
                    instance = np.mean(instance, axis=2)
                instance = cv2.resize(instance, (self.res, self.res))
                instance_masks.append(instance.astype(float) / 255.)
            out_sample['instances'] = instance_masks
        return out_sample

    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        return self.samples[idx]
