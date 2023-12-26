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

import OpenEXR, Imath
import os
import numpy as np
from tqdm import tqdm

def load_blender_exr(path, size, layers):
    data = OpenEXR.InputFile(path)
    px = Imath.PixelType(Imath.PixelType.HALF)
    out = {}

    if 'Image' in layers:
        image = []
        for channel in 'RGB':
            image_channel = 'Image.' + channel
            tmp = np.frombuffer(data.channel(image_channel, px), dtype=np.float16)
            image.append(tmp.reshape(size))
        image = np.stack(image, -1)
        image = np.maximum(0, image).astype('float16')
        out['Image'] = image

    if 'Direct' in layers:
        direct = []
        for channel in 'RGB':
            direct_channel = 'Diffuse Direct.' + channel
            tmp = np.frombuffer(data.channel(direct_channel, px), dtype=np.float16)
            direct.append(tmp.reshape(size))
        direct = np.stack(direct, -1)
        direct = np.maximum(0, direct).astype('float16')
        out['Direct'] = direct

    if 'Indirect' in layers:
        indirect = []
        for channel in 'RGB':
            indirect_channel = 'Diffuse Indirect.' + channel
            tmp = np.frombuffer(data.channel(indirect_channel, px), dtype=np.float16)
            indirect.append(tmp.reshape(size))
        indirect = np.stack(indirect, -1)
        indirect = np.maximum(0, indirect).astype('float16')
        out['Indirect'] = indirect

    if 'Albedo' in layers:
        color = []
        for channel in 'RGB':
            color_channel = 'Diffuse Albedo.' + channel
            tmp = np.frombuffer(data.channel(color_channel, px), dtype=np.float16)
            color.append(tmp.reshape(size))
        color = np.stack(color, -1)
        color = np.maximum(0, color).astype('float16')
        out['Albedo'] = color

    if 'Shadow' in layers:
        shadow = []
        tmp = np.frombuffer(data.channel('Image.A', px), dtype=np.float16)
        shadow = np.maximum(0, tmp.reshape(size)).astype('float16')
        out['Shadow'] = shadow

    if 'Position' in layers:
        positions = []
        for channel in 'XYZ':
            positions_channel = '3D Position.' + channel
            tmp = np.frombuffer(data.channel(positions_channel, px), dtype=np.float16)
            positions.append(tmp.reshape(size))
        positions = np.stack(positions, -1).astype('float16')
        out['Position'] = positions

    if 'Mask' in layers:
        mask = []
        tmp = np.frombuffer(data.channel('Mask.V', px), dtype=np.float16)
        mask = np.maximum(0, tmp.reshape(size)).astype('float16')
        out['Mask'] = mask

    return out


def load_blender_sample(folder, id, out):
    metadata = np.load(os.path.join(folder, str(id) + '_metadata.npz'))

    paths = [folder + str(id) + '_input.exr0001',
             folder + str(id) + '_input_shadows.exr0001',
             folder + str(id) + '_gt_sun.exr0001',
             folder + str(id) + '_gt_sky.exr0001',
             folder + str(id) + '_gt_ground.exr0001',
             folder + str(id) + '_obj.exr0001',
             folder + str(id) + '_obj_shadow.exr0001']

    names = ['Input',
             'Input Shadows',
             'GT Sun',
             'GT Sky',
             'GT Ground',
             'Obj',
             'Obj Shadow']

    layers = [['Image', 'Mask'],
              ['Shadow'],
              ['Image', 'Direct', 'Indirect'],
              ['Image', 'Direct', 'Indirect', 'Albedo', 'Position', 'Mask'],
              ['Image'],
              ['Image', 'Direct', 'Indirect'],
              ['Shadow']]

    files = {}
    for (name, path, layer) in zip(names, paths, layers):
        files[name] = load_blender_exr(path, metadata['resolution'], layer)

    gt = files['GT Sun']['Image'] + files['GT Sky']['Image']

    gt_indirect = files['GT Sun']['Indirect'] + files['GT Sky']['Direct'] + \
                  files['GT Sky']['Indirect']

    obj = (gt_indirect + files['Obj']['Direct']) * files['GT Sky']['Albedo'] * \
          np.repeat(files['GT Sky']['Mask'].reshape((metadata['resolution'][0],
                                                     metadata['resolution'][1],
                                                     1)), 3, axis=2)

    camera_matrix =  metadata['camera_intrinsic'] @ metadata['camera_extrinsic']

    input_shadows = files['Input Shadows']['Shadow'] * files['Input']['Mask']

    np.savez(os.path.join(folder, out, 'sample_' + str(id) + '.npz'),
             input =         files['Input']['Image'],
             input_shadows = input_shadows,
             ground_mask =   files['Input']['Mask'],
             gt_ground =     files['GT Ground']['Image'],
             gt =            gt,
             gt_direct =     files['GT Sun']['Direct'],
             gt_indirect =   gt_indirect,
             albedo =        files['GT Sky']['Albedo'],
             position =      files['GT Sky']['Position'],
             obj_mask =      files['GT Sky']['Mask'],
             obj =           obj,
             obj_direct =    files['Obj']['Direct'],
             shadow_mask =   files['Obj Shadow']['Shadow'],
             resolution =    metadata['resolution'],
             camera_matrix = camera_matrix,
             zenith =        metadata['zenith'],
             azimuth =       metadata['azimuth'])

DATASET = 'output_directory_path'
OUT = 'network_samples'
amount = 3600

for idx in tqdm(range(amount)):
    load_blender_sample(DATASET, idx, OUT)
