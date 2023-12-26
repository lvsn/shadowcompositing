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
from imageio import v2 as iio
from tqdm import tqdm
from models.fixup import FixUpUnet

@torch.no_grad()
def main():
    # Whether to use CUDA or CPU
    DEVICE = 'cuda'
    # Resolution the network was trained at
    NET_RES = 128
    # Batch size to utilize for simultaneous patch harmonization
    BATCH_SIZE = 64
    # Stride for the sliding window
    STRIDE = 1

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

    # Loading pre-trained model
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model = FixUpUnet(in_res=NET_RES, detection_input=MTMT_INPUT_BASELINE).to(DEVICE)
    model.load_state_dict(checkpoint['g_model_state'])
    model.eval()

    # Loading background image and adding reflection padding to avoid vignetting
    full_img = iio.imread(BACKGROUND_PATH).astype(float)
    h_orig, w_orig, _ = full_img.shape
    full_img = np.pad(full_img, ((NET_RES, NET_RES), (NET_RES, NET_RES), (0, 0)), 'reflect')
    h, w, _ = full_img.shape

    # Setting the input mask as the entire image by default (easier to reuse pieces afterward)
    # Note: can replace with only a desired region to speed-up computation
    input_mask = np.ones(full_img.shape[:2])
    input_mask = (torch.tensor(input_mask, dtype=torch.float16)).to(DEVICE, non_blocking=True)

    # Adapting the input for the baseline that uses our network's gain with MTMT's detection
    if MTMT_INPUT_BASELINE:
        npz = np.load(MTMT_DET_PATH)
        input_shadows = np.clip(npz['det'].astype(float) / 255, 0, 1)
        input_shadows = np.pad(input_shadows, ((NET_RES, NET_RES), (NET_RES, NET_RES)), 'reflect')
        input_shadows = (torch.tensor(input_shadows, dtype=torch.float16)).to(DEVICE, non_blocking=True)

    # Pre-allocating the necessary arrays for the batch
    full_img = np.moveaxis((full_img[:, :, :3] / 255), -1, 0) ** 2.2
    full_img = (torch.tensor(full_img, dtype=torch.float16)).to(DEVICE, non_blocking=True)
    full_gain = torch.zeros((3, h, w), dtype=torch.float32).to(DEVICE, non_blocking=True)
    full_det = torch.zeros((h, w), dtype=torch.float32).to(DEVICE, non_blocking=True)
    len_batch = 0
    batch_full = False
    batch_channels = 5 if MTMT_INPUT_BASELINE else 4
    batch = torch.ones((BATCH_SIZE, batch_channels, NET_RES, NET_RES),
                       dtype=torch.float16).to(DEVICE, non_blocking=True)
    batch_idx = np.zeros((BATCH_SIZE, 2), dtype=int)

    # Sliding window
    for i in tqdm(range(0, h - NET_RES + STRIDE + 1, STRIDE)):
        for j in range(0, w - NET_RES + STRIDE + 1, STRIDE):

            # Accumulating patches in the batch
            # Note: this needs to be done online otherwise becomes unfeasible memory-wise
            if j < (NET_RES + w_orig) and i < (NET_RES + h_orig):
                batch[len_batch, :3] = full_img[:, i:(i + NET_RES), j:(j + NET_RES)]
                batch[len_batch, 3] = input_mask[i:(i + NET_RES), j:(j + NET_RES)]
                if MTMT_INPUT_BASELINE:
                    batch[len_batch, 4] = input_shadows[i:(i + NET_RES), j:(j + NET_RES)]
                batch_idx[len_batch] = (i, j)
                len_batch += 1
                if len_batch == BATCH_SIZE:
                    batch_full = True
            ended = i >= (NET_RES + h_orig)

            # Processing a full batch
            if batch_full or ended:
                prev_idx = batch_idx[:len_batch].copy()

                # Inference
                with torch.autocast('cuda'):
                    x = model(2 * batch[:len_batch] - 1).detach()

                # Resetting batch
                len_batch = 0
                batch_full = False

                # Accumulating outputs
                for idx, sample in enumerate(x):
                    sample_i = prev_idx[idx, 0]
                    sample_j = prev_idx[idx, 1]
                    h_end = sample_i + NET_RES
                    w_end = sample_j + NET_RES
                    full_gain[:, sample_i:h_end, sample_j:w_end] += sample[0:3]
                    full_det[sample_i:h_end, sample_j:w_end] += sample[3]
                if ended:
                    break
            if ended:
                break

    # Processing accumulated outputs to obtain average of patches
    full_gain = full_gain.cpu().numpy()
    full_det = full_det.cpu().numpy()
    evals = (NET_RES / STRIDE) * (NET_RES / STRIDE) # Number of evaluations per pixel
    full_gain /= evals
    full_det /= evals
    full_gain = np.moveaxis(full_gain[:, NET_RES:NET_RES+h_orig, NET_RES:NET_RES+w_orig], 0, -1)
    full_det = full_det[NET_RES:NET_RES+h_orig, NET_RES:NET_RES+w_orig].reshape((h_orig, w_orig, 1)).repeat(3, axis=2)

    # Writing result
    np.savez(OUT_PATH, gain=full_gain, det=full_det)

if __name__ == '__main__':
    main()
