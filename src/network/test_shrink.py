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
import numpy as np
import cv2 as cv
from imageio import v2 as iio
from models.fixup import FixUpUnet

@torch.no_grad()
def main():
    # Whether to use CUDA or CPU
    DEVICE = 'cuda'
    # Resolution the network was trained at
    NET_RES = 128
    # Interpolation algorithm to utilize when resizing the inputs
    INTERP = cv.INTER_LINEAR_EXACT

    # Whether the standard network is being used
    # (or if it is our network's gain with MTMT's detection as input)
    MTMT_INPUT_BASELINE = False
    # Path to the MTMT detection (refer to example)
    MTMT_DET_PATH = 'your_path_here'

    # Path to the network's pre-trained weights
    CHECKPOINT_PATH = 'your_path_here'
    # Path to save the network's output
    OUT_PATH = 'your_path_here'
    # Path to the background image to be used as input
    BACKGROUND_PATH = 'your_path_here'
    # Path to the mask containing the pixels to harmonize
    MASK_PATH = 'your_path_here'

    # Loading the pre-trained model
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model = FixUpUnet(in_res=NET_RES, detection_input=MTMT_INPUT_BASELINE).to(DEVICE)
    model.load_state_dict(checkpoint['g_model_state'])
    model.eval()

    # Loading the background image
    full_img = iio.imread(BACKGROUND_PATH).astype(float)
    h_orig, w_orig, _ = full_img.shape
    full_img = cv.resize(full_img[:, :, :3], (NET_RES, NET_RES), interpolation=INTERP)
    full_img = (np.moveaxis(full_img, -1, 0) / 255) ** 2.2

    # Loading the mask of the region to harmonize
    input_mask = iio.imread(MASK_PATH).astype(float)
    input_mask = cv.resize(input_mask[:, :], (NET_RES, NET_RES), interpolation=INTERP)
    input_mask = (input_mask / 255).reshape((1, NET_RES, NET_RES))

    # Adapting the input for the baseline that uses our network's gain with MTMT's detection
    if MTMT_INPUT_BASELINE:
        npz = np.load(MTMT_DET_PATH)
        input_shadows = npz['det'].astype(float)
        input_shadows = cv.resize(input_shadows, (NET_RES, NET_RES), interpolation=INTERP).reshape((1, NET_RES, NET_RES))
        input_shadows = np.clip(input_shadows / 255, 0, 1)
        input = torch.tensor(np.concatenate([full_img, input_mask, input_shadows], axis=0)).float()
    else:
        input = torch.tensor(np.concatenate([full_img, input_mask], axis=0)).float()

    # Inference
    input = (2 * input.unsqueeze(0) - 1).to(DEVICE)
    x = model(input).detach().cpu()
    gain = np.moveaxis(x[0, 0:3].numpy(), 0, -1).astype(float)
    det = np.moveaxis(x[0, 3].repeat(3, 1, 1).numpy(), 0, -1).astype(float)

    # Saving results
    np.savez(OUT_PATH, gain=gain, det=det, res=(h_orig, w_orig))

if __name__ == '__main__':
    main()
