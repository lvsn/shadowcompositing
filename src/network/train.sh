#! /bin/bash
# Shadow Harmonization for Realisitc Compositing (c)
# by Lucas Valença, Jinsong Zhang, Michaël Gharbi,
# Yannick Hold-Geoffroy and Jean-François Lalonde.
#
# Developed at Université Laval in collaboration with Adobe, for more
# details please see <https://lvsn.github.io/shadowcompositing/>.
#
# Work published at ACM SIGGRAPH Asia 2023. Full open access at the ACM
# Digital Library, see <https://dl.acm.org/doi/10.1145/3610548.3618227>.
#
# This code is licensed under a Creative Commons
# Attribution-NonCommercial 4.0 International License.
#
# You should have received a copy of the license along with this
# work. If not, see <http://creativecommons.org/licenses/by-nc/4.0/>.

experiment_tag='debug'
device='cuda'

seed=42
reproducible=false
debug_dataset=true

use_synthetic=true
use_detection=true
use_srd=true

use_all_augmentations=true
use_ess_augmentation=false
use_nc_augmentation=false
use_ni_augmentation=false

use_ground_loss=true
use_direct_prediction=false
use_detection_input=false

epochs=3000
batch_size=12
in_size=128

learning_rate_g=0.0001
learning_rate_d=0.0001
adam_beta_1=0.9
adam_beta_2=0.999
train_val_split=0.9

DATASET="your_path_here"
city_dataset_dir="${DATASET}/datasets/CityShadows/"
istd_dataset_dir="${DATASET}/datasets/AISTD/"
srd_dataset_dir="${DATASET}/datasets/ASRD/"
desoba_dataset_dir="${DATASET}/datasets/ADESOBA/"
sbu_dataset_dir="${DATASET}/datasets/SBU/"

checkpoint_interval=2
checkpoint_dir='./checkpoints/'

checkpoint_restart=false
checkpoint_tag='tag_here'
checkpoint_epoch='99'

gan=true
gan_loss_weight=1
ground_loss_weight=1e2
shadow_loss_weight=3e1

arguments="--experiment_tag $experiment_tag --device $device \
           --seed $seed \
           --epochs $epochs --checkpoint_dir $checkpoint_dir \
           --checkpoint_interval $checkpoint_interval \
           --checkpoint_tag $checkpoint_tag \
           --checkpoint_epoch $checkpoint_epoch \
           --batch_size $batch_size --learning_rate_g $learning_rate_g \
           --learning_rate_d $learning_rate_d \
           --adam_betas $adam_beta_1 $adam_beta_2 --in_res $in_size \
           --city_dataset_dir $city_dataset_dir \
           --istd_dataset_dir $istd_dataset_dir \
           --srd_dataset_dir $srd_dataset_dir \
           --desoba_dataset_dir $desoba_dataset_dir \
           --sbu_dataset_dir $sbu_dataset_dir \
           --train_val_split $train_val_split \
           --gan_loss_weight $gan_loss_weight \
           --ground_loss_weight $ground_loss_weight \
           --shadow_loss_weight $shadow_loss_weight"

booleans=( reproducible gan checkpoint_restart debug_dataset \
           use_synthetic use_detection use_srd use_all_augmentations \
           use_ess_augmentation use_nc_augmentation use_ni_augmentation \
           use_direct_prediction use_detection_input use_ground_loss )

for arg in "${booleans[@]}"; do
    if [ "${!arg}" = true ]; then
        arguments="${arguments} --$arg";
    fi
done

python train.py $arguments
