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
from torch.utils.data.dataset import Dataset

# Synthetic path-traced dataset rendered on Blender
# Used by the Dataset class
class CityShadowsDataset(Dataset):
    def __init__(self, root_dir, in_res, train=True, debug=False):
        self.samples = []

        # Preparing filenames
        root = root_dir
        subset = 'train' if train else 'test'
        root = os.path.join(root, subset)
        filenames = os.listdir(root)
        if debug:
            filenames = filenames[:50]
        self.len = len(filenames)
        self.res = in_res

        # Loading samples
        for f in tqdm(filenames, total=self.len, leave=True, colour='blue',
                      desc='Loading Blender dataset: ', ncols=90):
            filepath = os.path.join(root, f)
            self.samples.append(self.prepare_sample(filepath))

    # Processing samples
    def prepare_sample(self, filepath):
        sample = np.load(filepath)

        # Shadows the virtual object is casting (network input mask to harmonize)
        mask = sample['shadow_mask']

        # Background image
        rgb_input = sample['input']
        in_expo = np.clip(np.percentile(rgb_input, 97), 1e-5, 1e5)
        rgb_input = np.clip(rgb_input / in_expo, 0, 1)

        # Harmonized target ground truth
        rgb_target = np.clip(sample['gt_ground'] / in_expo, 0, 1)

        # Ground truth shadows in the background image (shadow detection target)
        shadow_target = sample['input_shadows']

        # Output dictionary
        out_sample = {'input': rgb_input.astype(float),
                      'gt': rgb_target.astype(float),
                      'input_shadows': shadow_target.astype(float),
                      'mask': mask.astype(float)}
        return out_sample

    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        return self.samples[idx]
