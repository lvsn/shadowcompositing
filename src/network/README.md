# Network
This README assumes the user is running the provided Docker image (see our [Docker README](https://github.com/lvsn/shadowcompositing/blob/main/src/docker/README.md)) and that the necessary assets have been unpacked in the right relative locations (see the Assets section of our [main README](https://github.com/lvsn/shadowcompositing/blob/main/README.md)).

Included with our assets are the following pre-trained models:
- `main`: our proposed model as described in the main paper. This model can be used to most accurately reproduce our paper results.
- `main_alt`: same as above, but trained with a slightly different chance of each augmentation type happening (more inclined toward *subset mask blending*). Depending on the background image used (considering the vast range of possible outdoor scenes found online), this model may perform slightly better or worse.
- `gain_net`: baseline described in the paper which takes shadow detections as input. Can be used with MTMT's outputs (preferably without thresholding).
- `comp_net`: baseline described in the paper which directly outputs a harmonized ground instead of predicting a gain map.

## Training
Our network can be trained using the `train.sh` script. Flags within the script can be set however the user wishes. For specifics, comments and notes within our scripts can be used as reference.

- The `train.py` script launches (or restarts) the training process and automatically stores Tensorboard logs in the `runs` folder.
- The optimization loop is launched from `fit.py` by `train.py`, guided by the functions in `loss.py`.
- The `dataset.py` script aggregates our multiple data sources, with `datasets/real_dataset.py` loading our real shadow removal and detection datasets individually, and `datasets/city_dataset.py` loading our CityShadows synthetic dataset. All online data augmentation code can be found in `dataset.py`. This augmentation snippet can also be used to create fixed-seed offline augmented datasets for tasks such as benchmarking.
- The main generator network is described in `models/fixup.py`. Our discriminator can be found in `models/unet.py`.

## Testing
To test our network, we include two input-output variations (as shown in the main paper):

- `test_shrink.py` shrinks the input to the network's expected resolution, then stretches the network's output back to the original size. This is perhaps the most common approach for shadow detectors and runs almost instantaneously, however, it can lead to significant loss of detail and visible double shadow overlaps at soft boundaries.

- `test_pbla.py` utilizes our proposed sliding window algorithm, dubbed *Patch-Based Local Averaging*, or *PBLA* (see the main paper for details). Runtime varies significantly depending on the `stride` chosen. A lower stride will provide better results (finer details and smoother boundaries), but will take longer to compute.

For our ground-specific shadow compositing goal, PBLA tends to perform better due to the higher level of detail and soft boundary estimation, especially in everyday images and at higher resolutions.
