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
import cv2 as cv
import Imath
import OpenEXR
import imageio as iio

EPS = np.finfo(np.float16).eps

# Function to warp the shadows of the foreground virtual object render
def numpy_warp(image, xyz, camera_matrix, zenith, azimuth, resolution_x, resolution_y=None):
    if resolution_y == None:
        resolution_y = resolution_x

    # Magic number for the sun radius, anything far-enough away is fine
    radius = 100

    # Converting from spherical to cartesian coordinates
    sun = np.array([radius * np.sin(zenith) * np.cos(azimuth),
                    radius * np.sin(zenith) * np.sin(azimuth),
                    radius * np.cos(zenith)])

    # Vectorizing directional vectors per pixel (sun direction, ground plane point and normal)
    virtual_dir = np.ones((resolution_y, resolution_x, 3)) * -sun
    plane_pt = np.zeros((resolution_y, resolution_x, 3))
    plane_nor = np.ones((resolution_y, resolution_x, 3)) * np.array([0, 0, 1])

    # Intersecting vectors from the plane into the object's geometry
    t0 = (plane_pt - xyz)
    t1 = t0[:, :, 0] * plane_nor[:, :, 0] + \
         t0[:, :, 1] * plane_nor[:, :, 1] + \
         t0[:, :, 2] * plane_nor[:, :, 2]
    t2 = virtual_dir[:, :, 0] * plane_nor[:, :, 0] + \
         virtual_dir[:, :, 1] * plane_nor[:, :, 1] + \
         virtual_dir[:, :, 2] * plane_nor[:, :, 2]
    t2[t2 == 0] = -1
    t = t1 / t2
    t_3d = np.repeat(t.reshape((resolution_y, resolution_x, 1)), 3, axis=2)
    intersections = np.ones((resolution_y, resolution_x, 4))
    intersections[:, :, 0:3] = xyz
    intersections[t > 0, 0:3] = xyz[t > 0] + virtual_dir[t > 0] * t_3d[t > 0]
    intersections[:, :, 2] = 0
    intersections = intersections.reshape(resolution_y * resolution_x, 4)
    intersections = np.swapaxes(intersections, 0, 1)

    # Bringing intersections to camera space
    inter_2d = camera_matrix @ intersections
    inter_2d[0, :] /= inter_2d[2, :]
    inter_2d[1, :] /= inter_2d[2, :]
    inter_2d = np.swapaxes(inter_2d, 0, 1)
    inter_2d = inter_2d.reshape(resolution_y, resolution_x, 3)
    x = inter_2d[:, :, 1]
    y = inter_2d[:, :, 0]
    x[x > (resolution_y - 1)] = 0
    x[x < 0] = 0
    y[y > (resolution_x - 1)] = 0
    y[y < 0] = 0
    warp_tmp = np.zeros((resolution_y, resolution_x, 3))
    warp_tmp[t <= 0] = image[t <= 0]

    # Bilinear interpolation on ground pixels to avoid jagged edge artifacts
    x1 = x.astype(int)
    y1 = y.astype(int)
    x2 = x1 + 1
    y2 = y1 + 1
    p11 = (x1, y1)
    p12 = (x1, y2)
    p21 = (x2, y1)
    p22 = (x2, y2)
    x = np.repeat(x.reshape((resolution_y, resolution_x, 1)), 3, axis=2)
    y = np.repeat(y.reshape((resolution_y, resolution_x, 1)), 3, axis=2)
    x1 = np.floor(x).astype(int)
    y1 = np.floor(y).astype(int)
    x2 = x1 + 1
    y2 = y1 + 1
    fy1 = ((x2 - x) / (x2 - x1)) * image[p11[0], p11[1]] + \
          ((x - x1) / (x2 - x1)) * image[p21[0], p21[1]]
    fy2 = ((x2 - x) / (x2 - x1)) * image[p12[0], p12[1]] + \
          ((x - x1) / (x2 - x1)) * image[p22[0], p22[1]]
    warp_tmp[t > 0] = (y2 - y)[t > 0] * fy1[t > 0] + \
                      (y - y1)[t > 0] * fy2[t > 0]
    return warp_tmp


# Refine the gain map according to the mean of overlapping shadow regions
def refine_direct(bg_img, gain_map, detection, mask):
    rgb_bg = (mask * detection) * bg_img
    rgb_fg = (((mask * (1 - detection)) * gain_map) * bg_img**2.2)**(1/2.2)

    # Gross per-channel code to enable vectorization without numerical issues
    # Red
    if (rgb_fg[:, :, 0][rgb_fg[:, :, 0] > 1e-1].shape[0] != 0) and (rgb_bg[:, :, 0][rgb_bg[:, :, 0] > 1e-1].shape[0] != 0):
        mean_bg_r = rgb_bg[:, :, 0][rgb_bg[:, :, 0] > 1e-1].mean()
        mean_fg_r = rgb_fg[:, :, 0][rgb_fg[:, :, 0] > 1e-1].mean()
        r_factor = (mean_bg_r / (mean_fg_r + np.finfo(float).eps)) ** 2.2
        gain_map[:, :, 0] = (gain_map * r_factor * (1 - detection) + gain_map * detection)[:, :, 0]

    # Green
    if (rgb_fg[:, :, 1][rgb_fg[:, :, 1] > 1e-1].shape[0] != 0) and (rgb_bg[:, :, 1][rgb_bg[:, :, 1] > 1e-1].shape[0] != 0):
        mean_bg_g = rgb_bg[:, :, 1][rgb_bg[:, :, 1] > 1e-1].mean()
        mean_fg_g = rgb_fg[:, :, 1][rgb_fg[:, :, 1] > 1e-1].mean()
        g_factor = (mean_bg_g / (mean_fg_g + np.finfo(float).eps)) ** 2.2
        gain_map[:, :, 1] = (gain_map * g_factor * (1 - detection) + gain_map * detection)[:, :, 1]

    # Blue
    if (rgb_fg[:, :, 2][rgb_fg[:, :, 2] > 1e-1].shape[0] != 0) and (rgb_bg[:, :, 2][rgb_bg[:, :, 2] > 1e-1].shape[0] != 0):
        mean_bg_b = rgb_bg[:, :, 2][rgb_bg[:, :, 2] > 1e-1].mean()
        mean_fg_b = rgb_fg[:, :, 2][rgb_fg[:, :, 2] > 1e-1].mean()
        b_factor = (mean_bg_b / (mean_fg_b + np.finfo(float).eps)) ** 2.2
        gain_map[:, :, 2] = (gain_map * b_factor * (1 - detection) + gain_map * detection)[:, :, 2]

    return gain_map


# Loading an EXR containing the virtual object render to composite
def load_blender_exr(path, size):
    data = OpenEXR.InputFile(path)
    px = Imath.PixelType(Imath.PixelType.FLOAT)
    out = {}

    # Direct sunlight render pass
    direct = []
    for channel in 'RGB':
        direct_channel = 'direct.' + channel
        tmp = np.frombuffer(data.channel(direct_channel, px), dtype=np.float32)
        direct.append(tmp.reshape(size))
    direct = np.stack(direct, -1)

    # Indirect light (all remaining light in the scene since it has a single light source)
    indirect = []
    for channel in 'RGB':
        indirect_channel = 'indirect.' + channel
        tmp = np.frombuffer(data.channel(indirect_channel, px), dtype=np.float32)
        indirect.append(tmp.reshape(size))
    indirect = np.stack(indirect, -1)

    # Albedo per pixel
    albedo = []
    for channel in 'RGB':
        albedo_channel = 'albedo.' + channel
        tmp = np.frombuffer(data.channel(albedo_channel, px), dtype=np.float32)
        albedo.append(tmp.reshape(size))
    albedo = np.stack(albedo, -1)

    # RGB render of the object
    combined = []
    for channel in 'RGB':
        combined_channel = 'combined.' + channel
        tmp = np.frombuffer(data.channel(combined_channel, px), dtype=np.float32)
        combined.append(tmp.reshape(size))
    combined = np.stack(combined, -1)

    # Alpha channel of the render
    tmp = np.frombuffer(data.channel('combined.A', px), dtype=np.float32)
    combined_alpha = tmp.reshape(size)

    # HDR shadow cast by the object
    shadow = []
    for channel in 'RGB':
        shadow_channel = 'shadow.' + channel
        tmp = np.frombuffer(data.channel(shadow_channel, px), dtype=np.float32)
        shadow.append(tmp.reshape(size))
    shadow = np.stack(shadow, -1)
    full_shadow = shadow.copy()

    # Clipped shadow
    shadow = np.clip(shadow, 0, 1)

    # Path-traced shadow mask, as cast by the object
    tmp = np.frombuffer(data.channel('shadow.A', px), dtype=np.float32)
    shadow_alpha = tmp.reshape(size)

    # Anti-aliased binary visibility mask
    tmp = np.frombuffer(data.channel('mask.V', px), dtype=np.float32)
    mask = tmp.reshape(size)

    # XYZ positions per pixel
    positions = []
    for channel in 'XYZ':
        positions_channel = 'position.' + channel
        tmp = np.frombuffer(data.channel(positions_channel, px), dtype=np.float32)
        positions.append(tmp.reshape(size))
    positions = np.stack(positions, -1).astype(np.float32)

    # Normal vectors per pixel (normalized)
    normals = []
    for channel in 'XYZ':
        normals_channel = 'normal.' + channel
        tmp = np.frombuffer(data.channel(normals_channel, px), dtype=np.float32)
        normals.append(tmp.reshape(size))
    normals = np.stack(normals, -1).astype(np.float32)

    # Dictionary of render layers
    out = {'direct': direct,
           'indirect': indirect,
           'albedo': albedo,
           'combined': combined,
           'combined_alpha': combined_alpha,
           'mask': mask,
           'position': positions,
           'normal': normals,
           'shadow': shadow,
           'shadow_alpha': shadow_alpha,
           'full_shadow': full_shadow}
    return out


# Multiplier for the warped shadow intensities to compensate for lim -> 1
LAMBDA = 1.1
# Dynamic exposure percentile for tone-mapping (all compositing is HDR)
# Note: for bright sunlight scenes, 99 looks better
EXPOSURE = 100

# Full path to the LDR background image
BACKGROUND = 'your_path_here'
# Full path to the EXR render layers containing the object to insert
# Note: see the Blender file and rendering script example for clarity
SUN_RENDER = 'your_path_here'
SKY_RENDER = 'your_path_here'
SUN_SHADOW_RENDER = 'your_path_here'
SKY_SHADOW_RENDER = 'your_path_here'
# Full path to the Blender metadata containing camera, geometry, and illumination data
# Note: see the Blender file and rendering script example for clarity
METADATA = 'your_path_here'

# Full path to the network output for the background image
# Note: see the PBLA detection script for clarity
NET_OUT = 'your_path_here'
# Full path to save our composite video
COMP_OUT = 'your_path_here'

# Loading network outputs
ours_out = np.load(NET_OUT)
det = ours_out['det']
gain = ours_out['gain']

# Loading metadata
metadata = np.load(METADATA)
camera_matrix =  metadata['camera_intrinsic'] @ metadata['camera_extrinsic']
res = metadata['resolution']
zenith = metadata['zenith'].astype(float)
azimuth = metadata['azimuth'].astype(float)

# Loading renders
sun = load_blender_exr(SUN_RENDER, (res[1], res[0]))
sky = load_blender_exr(SKY_RENDER, (res[1], res[0]))
sun_shadow = load_blender_exr(SUN_SHADOW_RENDER, (res[1], res[0]))
sky_shadow = load_blender_exr(SKY_SHADOW_RENDER, (res[1], res[0]))

# Loading background
bg = iio.v2.imread(BACKGROUND).astype(float) / 255.
bg = bg[:, :, :3]
bg_linear = bg ** 2.2

# Loading render layers
obj_shadow = 1 - sun_shadow['shadow']
obj_mask = np.repeat(sun['mask'].reshape((res[1], res[0], 1)), 3, axis=-1)
obj_mask /= np.percentile(obj_mask, 100)
xyz = sun['position']
albedo = sun['albedo']
direct = sun['direct']
direct[obj_mask < 0.999] = np.minimum(direct[obj_mask < 0.999],
                                      sun['combined'][obj_mask < 0.999])
indirect = (sun['indirect'] + sky['indirect'] + sky['direct'])

# Start video writer
cap = cv.VideoWriter(COMP_OUT, cv.VideoWriter_fourcc(*'MP4V'), 10.0, (bg_linear.shape[1], bg_linear.shape[0]))

# Horizontal direction limits (in pixels) for 300 frames
# Note: this works with a single render only, moving horizontally.
#       Otherwise, render each frame separately and use the normal compositing script
for j in range(300):
    if j < 50:
        i = -j
    elif j > 200:
        i -= 1
    else:
        i = j - 100

    # Refine the gain
    gain_original = gain.copy()
    gain = refine_direct(bg, gain.copy(), det, obj_shadow)

    # Warp the detection
    warp_det = np.roll(det, -i, axis=1)
    warp = numpy_warp((1 - np.clip(LAMBDA * warp_det, 1e-5, 1)), xyz, camera_matrix,
                      zenith, azimuth, res[0], res[1])

    # Composite the foreground
    fg_comp = albedo * (indirect + direct * warp)
    expo = np.percentile(fg_comp * obj_mask, EXPOSURE)

    # Roll the frame to the side by 1px
    fg_comp = np.roll(fg_comp, i, axis=1)
    shadow_bg = np.roll(obj_shadow, i, axis=1)

    # Ambient occlusion
    highlights = sky_shadow['full_shadow']
    highlights_an = np.roll(highlights, i, axis=1)

    # Composite the background
    bg_comp = (shadow_bg * bg_linear * gain + (1 - shadow_bg) * bg_linear) * highlights_an
    moving_mask = np.roll(obj_mask, i, axis=1)

    # Composite foreground and background together then tone-map
    comp = (moving_mask * fg_comp + (1 - moving_mask) * bg_comp * expo) / expo
    ours_comp = (np.clip(comp**(1./2.2), 0, 1) * 255).astype('uint8')

    # Write the current frame
    cap.write(cv.cvtColor(ours_comp, cv.COLOR_RGB2BGR))
cap.release()
