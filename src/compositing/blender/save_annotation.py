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
import bpy

OBJ = ['bunny', 'ball', 'umbrella', 'glasses'] # Example 1 (teaser)
#OBJ = ['bunny'] # Example 2 (gray ground)
OUT_PATH = 'output_directory_path_here' # e.g., 'C:\\Downloads\'
TAG = 'scene_tag_for_filenames_here' # e.g., 'teaser'


# Utility function to save the camera's metadata
def get_camera_matrices(cam, scene):
    render = scene.render
    trans = cam.location
    rot = cam.rotation_euler.to_matrix()

    # Obtaining the extrinsic parameter matrix for the camera
    cam2world = np.array([[-rot[0][0], rot[0][1], rot[0][2], trans[0]],
                          [-rot[1][0], rot[1][1], rot[1][2], trans[1]],
                          [-rot[2][0], rot[2][1], rot[2][2], trans[2]],
                          [0, 0, 0, 1]])
    extrinsic = np.linalg.inv(cam2world)

    # Obtaining the intrinsic parameter matrix for the camera
    scale = render.resolution_percentage / 100
    res = (scale * render.resolution_x, scale * render.resolution_y)
    size = (render.pixel_aspect_x * res[0], render.pixel_aspect_y * res[1])

    sensor = (cam.data.sensor_width, cam.data.sensor_height)
    sensor_fit = cam.data.sensor_fit
    if sensor_fit == 'AUTO':
        sensor_fit = 'HORIZONTAL' if size[0] >= size[1] else 'VERTICAL'

    aspect = render.pixel_aspect_y / render.pixel_aspect_x
    view_factor = res[0] if sensor_fit == 'HORIZONTAL' else (res[1] * aspect)
    sensor_size = sensor[0] if sensor_fit == 'HORIZONTAL' else sensor[1]
    pixel_size = sensor_size / cam.data.lens / view_factor

    fx = 1 / pixel_size
    fy = 1 / pixel_size / aspect
    cx = res[0] / 2 - cam.data.shift_x * view_factor
    cy = res[1] / 2 + cam.data.shift_y * view_factor / aspect

    intrinsic = np.array([[fx, 0, cx, 0],
                          [0, fy, cy, 0],
                          [0, 0, 1, 0]])

    return extrinsic, intrinsic


# Rendering and writing
def save_keyframe():
    base_path = os.path.join(OUT_PATH, TAG)

    # Identifying necessary objects to render
    camera_name = TAG + '.fspy'
    camera = bpy.data.objects[camera_name]
    objects = [bpy.data.objects[x] for x in OBJ]
    sun = bpy.data.objects['Sun']
    world = bpy.data.worlds['World']
    sky = world.node_tree.nodes['Sky Texture']
    output = world.node_tree.nodes['World Output']
    links = world.node_tree.links
    scene = bpy.data.scenes['Scene']
    nodes = scene.node_tree.nodes
    sun_props = scene.sun_pos_properties

    # Following prints are self-explanatory
    print('Writing Metadata')
    cam_extrinsic, cam_intrinsic = get_camera_matrices(camera, scene)
    res = (scene.render.resolution_x, scene.render.resolution_y)

    np.savez(base_path + '_metadata.npz',
             tag=TAG,
             resolution=res,
             camera_extrinsic=cam_extrinsic,
             camera_intrinsic=cam_intrinsic,
             zenith=(0.5 * np.pi - sun_props.hdr_elevation),
             azimuth=sun_props.hdr_azimuth,
             elevation=sun_props.hdr_elevation,
             sun_strength=sun.data.energy,
             sky_air=sky.air_density,
             sky_dust=sky.dust_density,
             sky_ozone=sky.ozone_density)

    print('[1/4] Rendering Sky')
    sun.hide_render = True
    nodes['File Output'].base_path = base_path + '_sky'
    nodes['Value'].outputs[0].default_value = 0
    bpy.ops.render.render()
    nodes['Value'].outputs[0].default_value = 1
    sun.hide_render = False

    print('[2/4] Rendering Sky Shadow')
    sun.hide_render = True
    for object in objects:
        object.visible_camera = False
    nodes['File Output'].base_path = base_path + '_sky_shadow'
    bpy.ops.render.render()
    for object in objects:
        object.visible_camera = True
    sun.hide_render = False

    print('[3/4] Rendering Sun')
    links.remove(links[0])
    nodes['Value'].outputs[0].default_value = 0
    nodes['File Output'].base_path = base_path + '_sun'
    bpy.ops.render.render()
    nodes['Value'].outputs[0].default_value = 1
    links.new(sky.outputs[0], output.inputs[0])

    print('[4/4] Rendering Sun Shadow')
    links.remove(links[0])
    for object in objects:
        object.visible_camera = False
    nodes['File Output'].base_path = base_path + '_sun_shadow'
    bpy.ops.render.render()
    for object in objects:
        object.visible_camera = True
    links.new(sky.outputs[0], output.inputs[0])

    print('DONE')


if __name__ == '__main__':
    save_keyframe()
