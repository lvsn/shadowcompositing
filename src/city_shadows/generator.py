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

import random, os, csv
from math import sin, cos, tan, floor, ceil, sqrt
import numpy as np
import bpy
from mathutils import Vector, Matrix


#------------------------------------ ARGUMENTS --------------------------------
USE_SEED = True
SEED = 5

UPDATE_BUILDINGS = True
WRITE_FRAMES = True
DEBUG_PRINT = False
RENDER_DEBUG_KEYFRAME = False
NUM_FRAMES = 4000
FRAME_OFFSET = 0

OUTPUT_PATH = 'your_path_here'
SKY_DATASET_PATH = 'your_path_here'
POLYHAVEN_DATASET_PATH = 'your_path_here'
POLYHAVEN_RESOLUTION = '2k'
TEX_FILENAMES = {'diffuse': 'diff',
                 'translucent': 'translucent',
                 'roughness': 'rough',
                 'specular': 'spec',
                 'normal': 'nor_gl',
                 'displacement': 'disp'}

#------------------------------ TUNABLE PARAMETERS -----------------------------
# Render
# ------
DENOISE = True
HALF_PRECISION_EXR = True
RESOLUTION = (128, 128)
MAX_BOUNCES = 4
MAX_SAMPLES = 64

# Sky
# ---
CHISQUARE_DF = 3

# Main Object
# -----------
OBJ_SCALE = (0.003, 0.008)

# Flying Occluders
# ----------------
OCCLUDER_AMOUNT = (20, 60)
OCCLUDER_SIZE = (0.001, 0.075)
OCCLUDER_DISTANCE = (0.5, 1)
OCCLUDER_PERTURBATION = (15, 15)

# Camera
# ------
CAMERA_RADIUS = 0.45
CAMERA_ZENITH = (60, 85)
CAMERA_PERTURBATION = (-10, 10)

# Ground
# ------
ROAD_PROBABILITY = 0
SIDEWALK_PROBABILITY = 0.5
ENABLE_SPECULARITIES = False
ENABLE_TRANSLUCENCY = False
ENABLE_DISPLACEMENT = True


#----------------------------------- SCENE -------------------------------------
# Assets
# ------
MAIN_OBJ = 'bunny'
CAMERA = 'Camera'
GROUND_PLANE = 'Ground'
SCENE = 'Scene'
WORLD = 'World'
CITY = 'City'
CITY_AXIS = 'City Axis'
TERRAIN = 'Terrain'
OCCLUDERS = 'Occluders'

# Constants
# ---------
EPSILON = np.finfo(float).eps
TWO_PI = 2 * np.pi
PI = np.pi
HALF_PI = np.pi / 2
INFINITY = np.inf


#----------------------------- MATH UTIL FUNCTIONS -----------------------------
def equals(x, y):
    return abs(x - y) <= EPSILON


def divisible(x, y, get_division=False):
    res = False
    div = x / y

    if equals((x + EPSILON) % y, 0):
        res = True
        div = (x + EPSILON) / y
    elif equals(x % y, 0):
        res = True

    return res, div if get_division else res


def spherical2cartesian(radius, zenith, azimuth):
    return Vector([radius * sin(zenith) * cos(azimuth),
                   radius * sin(zenith) * sin(azimuth),
                   radius * cos(zenith)])


def line_plane_intersection(line_pt, line_dir, plane_pt, plane_nor):
    if -EPSILON <= np.dot(line_dir, plane_nor) <= EPSILON:
        return None

    t = np.dot((plane_pt - line_pt), plane_nor) / np.dot(line_dir, plane_nor)
    if t < 0:
        return None
    return line_pt + line_dir * t


def ray_ray_intersection(p0, dir0, p1, dir1):
    dx = p1.x - p0.x
    dy = p1.y - p0.y
    det = dir1.x * dir0.y - dir1.y * dir0.x

    if det == 0: return None

    t0 = (dy * dir1.x - dx * dir1.y) / det
    t1 = (dy * dir0.x - dx * dir0.y) / det

    if t0 > 0 and t1 > 0:
        return t0, t1

    return None


def ray_line_intersection(r0, dir0, l1, dir1):
    dx = l1.x - r0.x
    dy = l1.y - r0.y
    det = dir1.x * dir0.y - dir1.y * dir0.x

    if det == 0: return None

    t0 = (dy * dir1.x - dx * dir1.y) / det

    if t0 > 0: return t0

    return None


def get_azimuth(x, y):
    if x > 0:
        return (np.arctan(y / x)) % TWO_PI
    if x < 0:
        if y >= 0:
            return (np.arctan(y / x) + PI) % TWO_PI
        else:
            return (np.arctan(y / x) - PI) % TWO_PI
    if equals(x, 0):
        if y > 0:
            return HALF_PI % TWO_PI
        elif y < 0:
            return -HALF_PI % TWO_PI
        else:
            return 0


def approx_sky_radius(obj_location, sun_zenith, est_zenith_offset):
    obj = obj_location
    d0 = sqrt(obj[0] * obj[0] + obj[1] * obj[1] + obj[2] * obj[2])

    th0 = sun_zenith
    th1 = th0 + est_zenith_offset

    el0 = HALF_PI - th0
    el1 = HALF_PI - th1

    h0 = tan(el0) * d0
    d1 = (d0 * h0) / ((tan(el1) * d0) - h0)
    h1 = tan(el1) * d1
    r = h1 / cos(th0)

    return r


def approx_zenith_offset(obj_location, sun_zenith, sky_radius):
    obj = obj_location
    d0 = sqrt(obj[0] * obj[0] + obj[1] * obj[1] + obj[2] * obj[2])

    th0 = sun_zenith
    el0 = HALF_PI - th0

    h0 = tan(el0) * d0
    h1 = cos(th0) * sky_radius

    d1 = ((h1 * d0) / h0) - d0
    el1 = np.arctan(h1 / d1)
    offset = el0 - el1

    return offset


def move_pivot(obj, keep_position=False, pivot=None, bb_base=True):
    if pivot is None:
        verts = [v[:] for v in obj.bound_box]
        origin = np.mean(verts, axis=0)
        if bb_base:
            origin[2] = np.min([v[2] for v in verts])
    else:
        origin = pivot

    obj.data.transform(Matrix.Translation(-origin))

    if keep_position:
        world = obj.matrix_world
        world.translation = world @ Matrix() @ Vector(origin)

    return origin


def get_camera_matrices(cam, scene):
    render = scene.render

    trans = cam.location
    rot = cam.rotation_euler.to_matrix()

    cam2world = np.array([[-rot[0][0], rot[0][1], rot[0][2], trans[0]],
                          [-rot[1][0], rot[1][1], rot[1][2], trans[1]],
                          [-rot[2][0], rot[2][1], rot[2][2], trans[2]],
                          [0, 0, 0, 1]])
    extrinsic = np.linalg.inv(cam2world)

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


#-------------------------- ASSET RECOVERY FUNCTIONS ---------------------------

def get_polyhaven_materials(range, enable_spec=True, enable_trans=True,
                            enable_displacement=True):
    polyhaven_names = []
    for material in bpy.data.materials:
        if 'polyhaven' not in material.name: continue

        nodes = material.node_tree.nodes
        links = material.node_tree.links

        if enable_spec:
            nodes['Principled BSDF'].inputs[7].default_value = 0.5
        else:
            nodes['Principled BSDF'].inputs[7].default_value = 0

        if 'Translucent Texture' in nodes:
            link_found = False
            for link in links:
                if link.from_node == nodes['Translucent Texture']:
                    link_found = True
                    if not enable_trans: links.remove(link)
                    break

            if not link_found and enable_trans:
                links.new(nodes['Translucent Texture'].outputs[0],
                          nodes['Principled BSDF'].inputs[1])

        polyhaven_names.append(material.name)

    return polyhaven_names


def get_city_names(city_name):
    roads = []
    crossings = []
    buildings = []
    water_towers = []

    for obj in bpy.data.collections[city_name].objects:
            if 'SC road 1x1 1 straight ground' in obj.name:
                roads.append(obj.name)
            elif 'SC road 1x1 T crossing ground' in obj.name:
                crossings.append(obj.name)
            elif 'building' in obj.name:
                buildings.append(obj.name)
            elif 'water tower' in obj.name:
                water_towers.append(obj.name)

    return roads, crossings, buildings, water_towers


def get_skies(path, chisquare_df):
    skies = []

    with open(path + 'lm_annotations.csv', mode='r') as f:
        file = csv.reader(f)
        next(file)
        for line in file:
            if line[0] == 'Fail': continue

            gt_pano = line[1]
            sunsky_pano = line[2]
            sun_pano = sunsky_pano.replace('envmap.exr', 'sun.exr')
            sky_pano = sunsky_pano.replace('envmap.exr', 'sky.exr')
            zenith = float(line[3])
            azimuth = float(line[4])
            wsun = [float(line[5]), float(line[6]), float(line[7])]
            wsky = [float(line[8]), float(line[9]), float(line[10])]
            kappa = float(line[11])
            beta = float(line[12])
            turbidity = float(line[13])

            skies.append({'gt_pano': gt_pano,
                          'sunsky_pano': sunsky_pano,
                          'sun_pano': sun_pano,
                          'sky_pano': sky_pano,
                          'zenith': zenith,
                          'azimuth': azimuth,
                          'wsun': wsun,
                          'wsky': wsky,
                          'kappa': kappa,
                          'beta': beta,
                          'turbidity': turbidity})

    intensities = {}
    with open(path + 'annotations.csv', mode='r') as f:
        file = csv.reader(f)
        next(file)
        for line in file:
            intensities[line[0]] = float(line[1])

    for sky in skies:
        sky['intensity'] = intensities[sky['gt_pano']]

    skies = sorted(skies, key=lambda x : x['intensity'], reverse=True)
    dist = np.random.chisquare(chisquare_df, len(skies))
    dist = (np.sort(dist) / np.max(dist)) * (len(skies) - 1)
    sky_distribution = np.asarray(dist).astype(int)

    return skies, sky_distribution


#------------------------- CITY RAY CASTING FUNCTIONS --------------------------
def get_city_dims(city_objects, asphalt):
    range = [INFINITY, -INFINITY]

    for elem in city_objects:
        loc = bpy.data.objects[elem].location
        for i in [0, 1]:
            range[0] = loc[i] if loc[i] < range[0] else range[0]
            range[1] = loc[i] if loc[i] > range[1] else range[1]

    block_dim = round(bpy.data.objects[asphalt[0]].dimensions[0], 4)
    range = [range[0] - (block_dim / 2), range[1] + (block_dim / 2)]
    range = [round(range[0], 4), round(range[1], 4)]

    return range, block_dim


def get_grid_uvs(x, y, city_range, block_length):
    coords = []

    # Clip
    x = min(max(x - EPSILON, city_range[0]) + EPSILON, city_range[1])
    y = min(max(y - EPSILON, city_range[0]) + EPSILON, city_range[1])

    # Check if border
    x_border_l = equals(x, city_range[0])
    x_border_r = equals(x, city_range[1])

    y_border_t = equals(y, city_range[0])
    y_border_b = equals(y, city_range[1])

    # Begin range at origin
    x = x - city_range[0]
    y = y - city_range[0]

    # Check if is an edge and get UVs
    x_edge, x = divisible(x, block_length, get_division=True)
    y_edge, y = divisible(y, block_length, get_division=True)

    # List pairs in grid coords, for edges include both, consider borders
    if x_edge and y_edge and not x_border_l and not y_border_t:
        coords.append((floor(x - 1), floor(y - 1)))

    if x_edge and not x_border_l and not y_border_b:
        coords.append((floor(x - 1), floor(y)))

    if y_edge and not x_border_r and not y_border_t:
        coords.append((floor(x), floor(y - 1)))

    if not x_border_r and not y_border_b:
        coords.append((floor(x), floor(y)))

    return coords


def get_grid(city_range, block_length, elements):
    grid = []
    city_length = (city_range[1] - city_range[0])
    _, grid_size = divisible(city_length, block_length, get_division=True)
    grid_size = ceil(grid_size)

    for i in range(grid_size):
        grid.append([])
        for _ in range(grid_size):
            grid[i].append([])

    for elem in elements:
        loc = bpy.data.objects[elem].location
        coords = get_grid_uvs(loc[0], loc[1], city_range, block_length)
        grid[coords[-1][0]][coords[-1][1]].append(elem)
        u = (loc[0] - city_range[0]) / block_length
        v = (loc[1] - city_range[0]) / block_length

    return grid


def march_ray(obj_pt, sun_vec, ranges, block_len):
    _, x = divisible(obj_pt[0] - ranges[0], block_len, get_division=True)
    _, y = divisible(obj_pt[1] - ranges[0], block_len, get_division=True)
    x = (floor(x) * block_len) + ranges[0] # world coords for left cell edge
    y = (floor(y) * block_len) + ranges[0] # world coords for top cell edge

    tx = INFINITY
    ty = INFINITY
    stop = False
    (u, v) = obj_pt

    if (not equals(sun_vec[0], 0)):
        x = x + block_len if sun_vec[0] > 0 else x
        tx = (x - obj_pt[0]) / sun_vec[0]

    if (not equals(sun_vec[1], 0)):
        y = y + block_len if sun_vec[1] > 0 else y
        ty = (y - obj_pt[1]) / sun_vec[1]

    if tx is INFINITY and ty is INFINITY:
        stop = True
    else:
        if (tx <= ty):
            u = x
            v = obj_pt[1] + tx * sun_vec[1]
        else:
            u = obj_pt[0] + ty * sun_vec[0]
            v = y

        border_u = u <= ranges[0] or u >= ranges[1]
        border_v = v <= ranges[0] or v >= ranges[1]

        if (border_u and border_v) or (u, v) == obj_pt:
            stop = True

    return (u, v), stop


#--------------------------- FRAME UPDATE FUNCTIONS ----------------------------
def update_skies(path, skies, sky_distribution, world_name):
    sky_id = np.random.randint(low=0, high=len(skies))
    sky = skies[sky_distribution[sky_id]]

    for pano in ['gt', 'sun', 'sky', 'sunsky']:
        if (pano + '_panorama.exr') in bpy.data.images:
            bg = bpy.data.images[pano + '_panorama.exr']
            bpy.data.images.remove(bg, do_unlink=True)

    sun_img = bpy.data.images.load(path + sky['sun_pano'], check_existing=True)
    sun_img.name = 'sun_panorama.exr'
    sky_img = bpy.data.images.load(path + sky['sky_pano'], check_existing=True)
    sky_img.name = 'sky_panorama.exr'
    sunsky_img = bpy.data.images.load(path + sky['sunsky_pano'],
                                      check_existing=True)
    sunsky_img.name = 'sunsky_panorama.exr'
    gt_img = bpy.data.images.load(path + sky['gt_pano'], check_existing=True)
    gt_img.name = 'gt_panorama.exr'
    hdri = bpy.data.worlds[world_name].node_tree.nodes['Environment Texture']
    hdri.image = gt_img

    return sky


def update_object(obj_name, scales):
    obj = bpy.data.objects[obj_name]
    scale = scales[0] + np.random.rand(1) * (scales[1] - scales[0])
    obj.dimensions = ((obj.dimensions[0] / obj.scale[0]) * scale,
                      (obj.dimensions[1] / obj.scale[1]) * scale,
                      (obj.dimensions[2] / obj.scale[2]) * scale)
    bpy.context.view_layer.update()


def update_occluders(occluders_name, obj_name, sky, amounts, sizes, distances,
                     perturbations):

    occluders = bpy.data.collections[occluders_name].objects
    obj = bpy.data.objects[obj_name]

    for occluder in occluders:
        occluder.location[2] = -500
        occluder.visible_camera = False

    occluder_amount = np.random.randint(amounts[0], high=amounts[1])
    ids = np.random.choice(len(occluders), size=occluder_amount, replace=False)

    for id in ids:
        occluder = occluders[id]
        rand_dim = np.random.rand(3) * (sizes[1] - sizes[0]) + sizes[0]
        occluder.dimensions = Vector(rand_dim)

        tilt = np.random.randint(-perturbations[0], high=perturbations[0])
        pan = np.random.randint(-perturbations[1], high=perturbations[1])
        zenith = sky['zenith'] + np.deg2rad(tilt)
        azimuth = sky['azimuth'] + np.deg2rad(pan)

        sun = spherical2cartesian(1, zenith, azimuth)
        t = np.random.rand() * (distances[1] - distances[0]) + distances[0]

        occluder.location[0] = obj.location[0] + t * sun[0]
        occluder.location[1] = obj.location[1] + t * sun[1]
        occluder.location[2] = obj.location[2] + t * sun[2]

        rot = np.random.rand(3) * TWO_PI
        occluder.rotation_euler = rot


def update_camera(camera_name, obj_name, radius, zeniths, perturbations, sky):
    obj = bpy.data.objects[obj_name]
    cam = bpy.data.objects[camera_name]
    zenith = np.deg2rad(np.random.randint(low=zeniths[0], high=zeniths[1]))
    azimuth = np.deg2rad(np.random.randint(low=0, high=360))

    sphere_point = spherical2cartesian(radius, zenith, azimuth)
    cam.location = sphere_point + obj.location

    target = Vector((obj.location[0], obj.location[1],
                     (obj.location[2] + obj.dimensions[2] / 2)))

    rotation = (target - cam.location).to_track_quat('-Z', 'Y').to_euler()
    rotation = np.array([rotation[0], rotation[1], rotation[2]])

    tilt = np.random.randint(low=perturbations[0], high=perturbations[1])
    pan = np.random.randint(low=perturbations[0], high=perturbations[1])
    roll = np.random.randint(low=perturbations[0], high=perturbations[1])
    perturbation = np.array([np.deg2rad(x) for x in (tilt, pan, roll)])

    cam.rotation_euler = rotation + perturbation
    bpy.context.view_layer.update()

    ground = bpy.data.objects['Ground']
    ground_corners = [ground.matrix_world @ Vector(p) for p in ground.bound_box]
    min_z = np.min([p.z for p in ground_corners])
    bottom_bb = []
    dist_cam = []

    for id, p in enumerate(ground_corners):
        if p.z == min_z and p not in bottom_bb:
            bottom_bb.append(p)
            dist_cam.append(np.sqrt(((cam.location.x - p.x)**2 + (cam.location.y - p.y)**2)))

    min_ground_x = np.min([p.x for p in bottom_bb])
    max_ground_x = np.max([p.x for p in bottom_bb])
    min_ground_y = np.min([p.y for p in bottom_bb])
    max_ground_y = np.max([p.y for p in bottom_bb])

    ground_corners = [Vector((min_ground_x, min_ground_y, 0)),
                      Vector((max_ground_x, min_ground_y, 0)),
                      Vector((max_ground_x, max_ground_y, 0)),
                      Vector((min_ground_x, max_ground_y, 0))]

    ground_lines = [(0, 1), (1, 2), (2, 3), (3, 0)]

    selected_corners = []
    for id, corner in enumerate(ground_corners):
        if ((Vector((0, 0, 0)) - cam.location).normalized().dot((corner - cam.location).normalized()) > 0):
            selected_corners.append(id)

    selected_lines = []
    for id, line in enumerate(ground_lines):
        if id not in selected_lines and (line[0] in selected_corners or line[1] in selected_corners):
            selected_lines.append(id)

    frame = cam.data.view_frame()
    tr, br, bl, tl = [cam.matrix_world @ corner for corner in frame]
    vec_corners = [Vector((c - cam.location)).normalized() for c in [tr, br, bl, tl]]

    br = Vector(line_plane_intersection(cam.location, vec_corners[1], np.array([0, 0, 0]), np.array([0, 0, 1])))
    bl = Vector(line_plane_intersection(cam.location, vec_corners[2], np.array([0, 0, 0]), np.array([0, 0, 1])))

    # TODO: 10 magic number
    far_l = vec_corners[2] * 10 + cam.location
    bl_depth = sqrt((bl.x - cam.location.x)**2 + (bl.y - cam.location.y)**2 + (bl.z - cam.location.z)**2)
    up_pt = vec_corners[3] * bl_depth + cam.location
    frustum_plane_dir = (far_l - bl).normalized().cross((up_pt - bl).normalized())
    bl_ground = Vector((0, 0, 1)).cross(frustum_plane_dir).normalized()

    tl_intersections = []
    for line in selected_lines:
        l = ground_corners[ground_lines[line][0]]
        l_dir = Vector(ground_corners[ground_lines[line][1]] - ground_corners[ground_lines[line][0]]).normalized()
        t = ray_line_intersection(bl, bl_ground, l, l_dir)

        if t is not None:
            tl_intersections.append(bl + t * bl_ground)

    min_tl_dist = INFINITY
    for inter in tl_intersections:
        if inter is not None:
            dist = np.sqrt(((inter.x - bl.x)**2 + (inter.y - bl.y)**2 + (inter.z - bl.z)**2))
            if dist < min_tl_dist:
                min_tl_dist = dist
                tl = inter

    far_r = vec_corners[1] * 10 + cam.location
    br_depth = sqrt((br.x - cam.location.x)**2 + (br.y - cam.location.y)**2 + (br.z - cam.location.z)**2)
    up_pt = vec_corners[0] * br_depth + cam.location
    frustum_plane_dir = (far_r - br).normalized().cross((up_pt - br).normalized())
    br_ground = Vector((0, 0, 1)).cross(frustum_plane_dir).normalized()

    tr_intersections = []
    for line in selected_lines:
        l = ground_corners[ground_lines[line][0]]
        l_dir = Vector(ground_corners[ground_lines[line][1]] - ground_corners[ground_lines[line][0]]).normalized()
        t = ray_line_intersection(br, br_ground, l, l_dir)

        if t is not None:
            tr_intersections.append(br + t * br_ground)

    min_tr_dist = INFINITY
    for inter in tr_intersections:
        if inter is not None:
            dist = np.sqrt(((inter.x - br.x)**2 + (inter.y - br.y)**2 + (inter.z - br.z)**2))
            if dist < min_tr_dist:
                min_tr_dist = dist
                tr = inter

    cam_corners = [tr, br, bl, tl]
    sun = Vector(spherical2cartesian(1, sky['zenith'], sky['azimuth']))
    bb = [obj.matrix_world @ Vector(p) for p in obj.bound_box]
    max_z = np.max([p.z for p in bb])
    top_bb = []

    for pt in bb:
        if pt.z == max_z and pt not in top_bb:
            inter = Vector(line_plane_intersection(pt, -sun, np.array([0, 0, 0]), np.array([0, 0, 1])))
            if inter is not None:
                top_bb.append(inter)

    dist_sun = []

    for id, p in enumerate(top_bb):
        dist_sun.append(np.sqrt(((10*sun.x - p.x)**2 + (10*sun.y - p.y)**2)))

    sun_pts = np.argsort(dist_sun)
    sun_pts = [sun_pts[-1], sun_pts[-2]]
    frustum_lines = [(0, 1), (1, 2), (2, 3), (3, 0)]
    max_delta = 0
    tip = None
    line = None

    for l in frustum_lines:
        l0 = Vector((cam_corners[l[0]].x, cam_corners[l[0]].y, 0))
        dl = Vector((cam_corners[l[0]].x - cam_corners[l[1]].x,
                     cam_corners[l[0]].y - cam_corners[l[1]].y, 0)).normalized()

        for id in sun_pts:
            tip_dir = top_bb[id].normalized()
            t = ray_line_intersection(Vector((0, 0, 0)), tip_dir, l0, dl)

            if t is not None:
                lim_tip = t * tip_dir
                lim_dist = np.sqrt(lim_tip.x**2 + lim_tip.y**2 + lim_tip.z**2)
                tip_dist = np.sqrt(top_bb[id].x**2 + top_bb[id].y**2 + top_bb[id].z**2)

                delta = tip_dist - lim_dist
                if delta > max_delta:
                    max_delta = delta
                    tip = top_bb[id]
                    line = l

    if max_delta == 0:
        return

    l0 = Vector((cam_corners[line[0]].x, cam_corners[line[0]].y, 0))
    dl = Vector((cam_corners[line[0]].x - cam_corners[line[1]].x,
                 cam_corners[line[0]].y - cam_corners[line[1]].y, 0)).normalized()

    tip_dir = tip.normalized()

    det = dl.x * tip_dir.y - dl.y * tip_dir.x
    t = (l0.y * dl.x - l0.x * dl.y) / det
    limit = (0.95 * t) * tip_dir

    limit_radius = np.sqrt(limit.x**2 + limit.y**2 + limit.z**2)
    tip_radius = np.sqrt(tip.x**2 + tip.y**2 + tip.z**2)

    obj.dimensions = ((obj.dimensions[0] / obj.scale[0]) * ((obj.scale[0] * (limit_radius/tip_radius))),
                      (obj.dimensions[1] / obj.scale[1]) * ((obj.scale[1] * (limit_radius/tip_radius))),
                      (obj.dimensions[2] / obj.scale[2]) * ((obj.scale[2] * (limit_radius/tip_radius))))
    bpy.context.view_layer.update()


def toggle_white_ground(flag, ground):
    if len(ground.data.materials) == 0:
        return

    mix_factor = 0 if flag == True else 1
    material = ground.data.materials[0]
    tree = material.node_tree

    if material.use_nodes and 'White Mix' in tree.nodes:
        mix = tree.nodes['White Mix']
        mix.inputs[0].default_value = mix_factor

        white = tree.nodes['White Diffuse BSDF']

        if flag == True:
            for link in tree.links:
                if link.to_socket == white.inputs[2]:
                    tree.links.remove(link)
                if link.to_socket == white.inputs[1]:
                    tree.links.remove(link)


def toggle_white_diffuses(value):
    value = 0 if value else 1

    for material in bpy.data.materials:
        if material.use_nodes and 'White Mix' in material.node_tree.nodes:
            mix = material.node_tree.nodes['White Mix']
            mix.inputs[0].default_value = value


def update_buildings(obj_name, axis_name, sky, grid, ranges, block_len):
    obj = bpy.data.objects[obj_name].location
    offset = bpy.data.objects[axis_name].location
    orig = (obj[0] - offset[0], obj[1] - offset[1])

    sun = spherical2cartesian(1, sky['zenith'], sky['azimuth'])
    sun_vec = (sun[0], sun[1])

    sun_left = spherical2cartesian(1, sky['zenith'],
                                   (sky['azimuth'] + np.deg2rad(15)) % TWO_PI)
    sun_right = spherical2cartesian(1, sky['zenith'],
                                   (sky['azimuth'] - np.deg2rad(15)) % TWO_PI)
    sun_left = (sun_left[0], sun_left[1])
    sun_right = (sun_right[0], sun_right[1])

    hits = []

    ray = (orig[0], orig[1])
    stop = False
    while not stop:
        if DEBUG_PRINT:
            print('Front 0: ' + str(ray[0]) + ', ' + str(ray[1]))
        hits = hits + get_grid_uvs(ray[0], ray[1], ranges, block_len)
        ray = (ray[0] + sun_vec[0], ray[1] + sun_vec[1])
        stop = ray[0] < ranges[0] or ray[1] < ranges[0] or \
               ray[0] > ranges[1] or ray[1] > ranges[1]

    ray = (orig[0], orig[1])
    sun_vec = sun_left
    stop = False
    while not stop:
        if DEBUG_PRINT:
            print('Left 0: ' + str(ray[0]) + ', ' + str(ray[1]))
        hits = hits + get_grid_uvs(ray[0], ray[1], ranges, block_len)
        ray = (ray[0] + sun_vec[0], ray[1] + sun_vec[1])
        stop = ray[0] < ranges[0] or ray[1] < ranges[0] or \
               ray[0] > ranges[1] or ray[1] > ranges[1]

    ray = (orig[0], orig[1])
    sun_vec = sun_right
    stop = False
    while not stop:
        if DEBUG_PRINT:
            print('Right 0: ' + str(ray[0]) + ', ' + str(ray[1]))
        hits = hits + get_grid_uvs(ray[0], ray[1], ranges, block_len)
        ray = (ray[0] + sun_vec[0], ray[1] + sun_vec[1])
        stop = ray[0] < ranges[0] or ray[1] < ranges[0] or \
               ray[0] > ranges[1] or ray[1] > ranges[1]

    uvs = get_grid_uvs(orig[0], orig[1], ranges, block_len)[0]
    changes = []
    hits = set(hits)
    for hit in hits:
        r = sqrt((hit[0] - uvs[0]) ** 2 + (hit[1] - uvs[1]) ** 2) * block_len

        target_zenith = min(sky['zenith'] + np.deg2rad(30), np.deg2rad(89))
        target_height = tan((PI / 2) - target_zenith) * r
        target_height = target_height + bpy.data.objects[obj_name].location[2]
        target_height = max(target_height, 0)

        block_height = 0

        for element in grid[hit[0]][hit[1]]:
            elem = bpy.data.objects[element]

            origin = move_pivot(elem, keep_position=True)
            elem_height = elem.dimensions[2] + elem.location[2]
            move_pivot(elem, keep_position=True, pivot=-origin)

            if elem_height > block_height:
                block_height = elem_height

        if block_height > target_height:
            for element in grid[hit[0]][hit[1]]:
                elem = bpy.data.objects[element]
                changes.append({'name': element, 'z': elem.location[2]})
                delta = block_height - target_height
                if DEBUG_PRINT:
                    print('Flattening ' + element + ' by ' + str(delta))
                elem.location[2] = elem.location[2] - delta

    return changes


def update_surface(road_names, ground_name, axis_name, material_names,
                   road_prob, sidewalk_prob):

    road_id = np.random.randint(low=0, high=len(road_names))
    road_name = road_names[road_id]
    road_surface = bpy.data.objects[road_name]
    ground_plane = bpy.data.objects[ground_name]

    p_road = road_prob * 1.1 + np.random.rand()
    p_sidewalk = np.random.rand()

    axis = bpy.data.objects[axis_name]
    axis.location[0] = -road_surface.location[0]
    axis.location[1] = -road_surface.location[1]

    if p_road > road_prob:
        ground_plane.hide_viewport = False
        ground_plane.hide_render = False

        mat_id = np.random.randint(low=0, high=len(material_names))
        mat = bpy.data.materials[material_names[mat_id]]
        if len(ground_plane.data.materials) > 0:
            ground_plane.data.materials[0] = mat
        else:
            ground_plane.data.materials.append(mat)

        if p_sidewalk > sidewalk_prob:
            axis.location[2] = -road_surface.dimensions[2] - 0.00001
        else:
            axis.location[2] = (-road_surface.dimensions[2] / 2) - 0.00055

    else:
        ground_plane.hide_viewport = True
        ground_plane.hide_render = True
        axis.location[2] = -road_surface.dimensions[2] / 2

    return ground_plane if p_road > road_prob else road_surface

#----------------------------------- SET UP ------------------------------------
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    bpy.data.scenes['Scene'].cycles.seed = seed


def set_cycles(scene_name, max_bounces, max_samples, denoise, resolution):
    scene = bpy.data.scenes[scene_name]
    cycles = bpy.data.scenes[scene_name].cycles

    cycles.use_denoising = denoise
    cycles.device = 'GPU'
    cycles.denoiser = 'OPTIX'
    cycles.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
    cycles.max_bounces = max_bounces
    cycles.transparent_max_bounces = 4
    cycles.samples = max_samples
    cycles.tile_size = resolution[0]

    scene.render.use_file_extension = False
    scene.render.resolution_percentage = 100


def set_scene_settings(world_name, main_obj_name):
    mapping = bpy.data.worlds[world_name].node_tree.nodes['Mapping']
    mapping.inputs[3].default_value[1] = -1


def set_polyhaven_texture(tree, link, tex_type, tex_names, path, res, blend):
    link.from_node.name = tex_type + ' texture'

    img = link.from_node.image
    if img and img.name == 'polyhaven_' + blend + '_' + tex_type:
        return
    elif 'polyhaven_' + blend + '_' + tex_type in bpy.data.images:
        if img:
            bpy.data.images.remove(img, do_unlink=True)
        img = bpy.data.images['polyhaven_' + blend + '_' + tex_type]
        link.from_node.image = img
        return

    ext = 'exr' if '.exr' in img.name else 'png'
    bpy.data.images.remove(img, do_unlink=True)

    file_path = path + ext + '\\' + blend + '_'
    ext_path = '_' + res + '.' + ext
    full_path = file_path + tex_names[tex_type] + ext_path
    found = os.path.exists(full_path)

    if not found and tex_type == 'roughness':
        full_path = file_path + tex_names['specular'] + ext_path
        found = os.path.exists(full_path)

    if found:
        img = bpy.data.images.load(full_path, check_existing=True)
        img.name = 'polyhaven_' + blend + '_' + tex_type
        link.from_node.image = img
    else:
        link.from_node.image = None
        tree.links.remove(link)
        print(tex_type + ' not found for file ' + blend)


def set_polyhaven_materials(path, res, tex_names):
    blends = os.listdir(path + 'blend\\')

    for blend in blends:
        blend = blend.replace('_' + res + '.blend', '')
        name = blend + '_' + res + '.blend\\Material\\' + blend

        if 'polyhaven_' + blend in bpy.data.materials: continue

        found_diffuse = False
        for ext in ['png', 'exr']:
            diffuse_path = path + ext + '\\' + blend + '_' + \
                           tex_names['diffuse'] + '_' + res + '.' + ext
            found_diffuse = os.path.exists(diffuse_path)
            if found_diffuse: break

        if not found_diffuse: continue

        material_len = len(bpy.data.materials)
        bpy.ops.wm.append(filename=name, directory=(path + 'blend\\'))
        if len(bpy.data.materials) == material_len: continue

        tree = bpy.data.materials[blend].node_tree
        tree.nodes['Principled BSDF'].inputs[7].default_value = 0.5
        tree.nodes['Principled BSDF'].inputs[9].default_value = 0.5
        tree.nodes['Principled BSDF'].inputs[1].default_value = 0

        if 'Displacement' in tree.nodes:
            tree.nodes['Displacement'].inputs[1].default_value = 0

        for link in tree.links:
            if link.to_node == tree.nodes['Principled BSDF']:
                if link.to_socket == tree.nodes['Principled BSDF'].inputs[0]:
                    set_polyhaven_texture(tree, link, 'diffuse', tex_names,
                                          path, res, blend)
                elif link.to_socket == tree.nodes['Principled BSDF'].inputs[1]:
                    set_polyhaven_texture(tree, link, 'translucent', tex_names,
                                          path, res, blend)
                elif link.to_socket == tree.nodes['Principled BSDF'].inputs[9]:
                    set_polyhaven_texture(tree, link, 'roughness',
                                          tex_names, path, res, blend)

            elif link.to_node.type == 'NORMAL_MAP':
                set_polyhaven_texture(tree, link, 'normal',
                                      tex_names, path, res, blend)
            elif link.to_node.type == 'DISPLACEMENT':
                set_polyhaven_texture(tree, link, 'displacement',
                                      tex_names, path, res, blend)

        for node in tree.nodes:
            if node.type == 'TEX_IMAGE' and \
               node.name.replace(' texture', '') not in tex_names.keys():
                print('unindexed texture ' + node.name + ' found in ' + blend)

        bpy.data.materials[blend].name = 'polyhaven_' + blend


def set_white_diffuses():
    for material in bpy.data.materials:
        if material.use_nodes:
            tree = material.node_tree

            if 'White Diffuse BSDF' not in tree.nodes:
                white = tree.nodes.new('ShaderNodeBsdfDiffuse')
                white.name = 'White Diffuse BSDF'
                white.inputs[0].default_value = Vector((0.9, 0.9, 0.9, 1))

                white_mix = tree.nodes.new('ShaderNodeMixShader')
                white_mix.name = 'White Mix'
                white_mix.inputs[0].default_value = 1

                white.location = 350, 800
                white_mix.location = 700, 800

                out = None
                for node in tree.nodes:
                    if 'ShaderNodeOutputMaterial' == node.type or \
                       'OUTPUT_MATERIAL' == node.type:
                            out = node

                if out == None:
                    print(material.name)

                prior = None
                for link in tree.links:
                    if link.to_socket == out.inputs[0]:
                        prior = link.from_socket
                        tree.links.remove(link)
                        break

                tree.links.new(white.outputs[0], white_mix.inputs[1])
                tree.links.new(prior, white_mix.inputs[2])
                tree.links.new(white_mix.outputs[0], out.inputs[0])

                if 'roughness texture' in tree.nodes:
                    tree.links.new(tree.nodes['roughness texture'].outputs[0],
                                   white.inputs[1])

                for node in tree.nodes:
                    if 'NORMAL_MAP' == node.type:
                        node.name = 'Normal Map'
                        tree.links.new(node.outputs[0], white.inputs[2])
                        break

            for node in tree.nodes:
                    if 'NORMAL_MAP' == node.type:
                        node.name = 'Normal Map'


def fix_objects(main=None, axis=None, ground=None, range=None, block=None):
    if main:
        bpy.data.objects[main].location = Vector((0, 0, 0))
        move_pivot(bpy.data.objects[main])
        bpy.data.objects[main].pass_index = 1
    if axis:
        bpy.data.objects[axis].location = Vector((0, 0, 0))
    if ground:
        bpy.data.objects[ground].location = Vector((0, 0, 0))
        bpy.data.objects[ground].pass_index = 2

    bpy.context.view_layer.update()

def fix_buildings(grid, changes=None):
    if changes:
        for change in changes:
            bpy.data.objects[change['name']].location[2] = change['z']
            bpy.context.view_layer.update()
            if DEBUG_PRINT:
                print('Fixing ' + str(change['name']))

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            min_grid = np.inf
            for elem in grid[i][j]:
                elem = bpy.data.objects[elem]

                orig = move_pivot(elem, keep_position=True)
                offset = elem.location[2]
                if offset < min_grid:
                    min_grid = offset
                move_pivot(elem, keep_position=True, pivot=-orig)

            if min_grid < -EPSILON:
                if DEBUG_PRINT:
                    print('Fixing height at coords. (' + \
                          str(i) + ', ' + str(j) + '). Broken by ' + str(min_grid))

                for elem in grid[i][j]:
                    if DEBUG_PRINT:
                        print('Fixing ' + str(elem))
                    elem = bpy.data.objects[elem]
                    elem.location[2] = elem.location[2] - min_grid

    bpy.context.view_layer.update()


#------------------------------ FRAME GENERATION -------------------------------
def save_keyframe(id, path, scene_name, half_precision, resolution,
                  obj_name, occluders_name, city_name, terrain_name,
                  camera_name, world_name, surface, sky, render_full=False):

    occluders = bpy.data.collections[occluders_name]
    city = bpy.data.collections[city_name]
    terrain = bpy.data.collections[terrain_name]
    obj = bpy.data.objects[obj_name]
    camera = bpy.data.objects[camera_name]
    hdri = bpy.data.worlds[world_name].node_tree.nodes['Environment Texture']
    scene = bpy.data.scenes[scene_name]
    nodes = scene.node_tree.nodes
    view = scene.view_layers['ViewLayer']

    if id % 25 == 0: # Clearing cache
        scene.render.use_persistent_data = False
    else:
        scene.render.use_persistent_data = True
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    settings = scene.render.image_settings

    settings.file_format = 'OPEN_EXR_MULTILAYER'
    settings.color_depth = '16' if half_precision else '32'
    settings.exr_codec = 'ZIP'
    settings.color_mode = 'RGB'
    nodes['File Output'].format.color_mode = 'RGB'

    print('Writing Metadata')
    obj_extrinsic = np.asarray([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
    cam_extrinsic, cam_intrinsic = get_camera_matrices(camera, scene)

    np.savez(path + str(id) + '_metadata.npz',
             gt_pano=sky['gt_pano'], lm_pano=sky['sunsky_pano'],
             resolution=resolution, precision=settings.color_depth,
             obj_extrinsic=obj_extrinsic,
             camera_extrinsic = cam_extrinsic, camera_intrinsic=cam_intrinsic,
             zenith=sky['zenith'], azimuth=sky['azimuth'], wsun=sky['wsun'],
             wsky=sky['wsky'], kappa=sky['kappa'], beta=sky['beta'],
             turbidity=sky['turbidity'])

    view.use_pass_normal = False
    view.use_pass_z = False
    view.use_pass_diffuse_direct = False
    view.use_pass_diffuse_indirect = False
    view.use_pass_diffuse_color = False
    view.use_pass_position = False

    city.hide_render = False
    occluders.hide_render = False
    terrain.hide_render = False
    obj.hide_render = False
    obj.visible_camera = True
    obj.visible_diffuse = True

    print('(1/7) Rendering Scene Shadow Catcher')
    view.use_pass_object_index = False
    view.use_pass_combined = True
    obj.hide_render = True
    scene.render.film_transparent = True
    surface.is_shadow_catcher = True
    settings.color_mode = 'RGBA'
    nodes['File Output'].format.color_mode = 'RGBA'
    hdri.image = bpy.data.images['sun_panorama.exr']
    nodes['File Output'].base_path = path + str(id) + '_input_shadows.exr'
    bpy.ops.render.render()
    scene.render.film_transparent = False
    surface.is_shadow_catcher = False

    print('(2/7) Rendering Input')
    settings.color_mode = 'RGB'
    nodes['File Output'].format.color_mode = 'RGB'
    view.use_pass_object_index = True
    nodes['ID Mask'].index = 2
    hdri.image = bpy.data.images['sunsky_panorama.exr']
    nodes['File Output'].base_path = path + str(id) + '_input.exr'
    bpy.ops.render.render()

    print('(3/7) Rendering GT Ground')
    obj.hide_render = False
    obj.visible_camera = False
    obj.visible_diffuse = False
    nodes['File Output'].base_path = path + str(id) + '_gt_ground.exr'
    bpy.ops.render.render()

    print('(4/7) Rendering GT Object Sky')
    view.use_pass_object_index = True
    view.use_pass_diffuse_direct = True
    view.use_pass_diffuse_indirect = True
    view.use_pass_diffuse_color = True
    view.use_pass_position = True
    nodes['ID Mask'].index = 1
    obj.visible_camera = True
    obj.visible_diffuse = True
    hdri.image = bpy.data.images['sky_panorama.exr']
    nodes['File Output'].base_path = path + str(id) + '_gt_sky.exr'
    bpy.ops.render.render()

    print('(5/7) Rendering GT Object Sun')
    view.use_pass_object_index = False
    view.use_pass_diffuse_color = False
    view.use_pass_position = False
    hdri.image = bpy.data.images['sun_panorama.exr']
    nodes['File Output'].base_path = path + str(id) + '_gt_sun.exr'
    bpy.ops.render.render()

    print('(6/7) Rendering Virtual Object Sun')
    view.use_pass_combined = False
    city.hide_render = True
    occluders.hide_render = True
    terrain.hide_render = True
    nodes['File Output'].base_path = path + str(id) + '_obj.exr'
    bpy.ops.render.render()

    print('(7/7) Rendering Virtual Shadow Catcher')
    view.use_pass_combined = True
    view.use_pass_diffuse_direct = False
    scene.render.film_transparent = True
    surface.is_shadow_catcher = True
    settings.color_mode = 'RGBA'
    nodes['File Output'].format.color_mode = 'RGBA'
    obj.visible_camera = False
    obj.visible_diffuse = False
    nodes['File Output'].base_path = path + str(id) + '_obj_shadow.exr'
    bpy.ops.render.render()

    scene.render.film_transparent = False
    surface.is_shadow_catcher = False


#------------------------------------ MAIN -------------------------------------
def main():
    if USE_SEED: set_seeds(SEED)

    print('Initial Setup')
    set_cycles(SCENE, MAX_BOUNCES, MAX_SAMPLES, DENOISE, RESOLUTION)
    set_scene_settings(WORLD, MAIN_OBJ)

    print('Linking Polyhaven Materials')
    set_polyhaven_materials(POLYHAVEN_DATASET_PATH,
                            POLYHAVEN_RESOLUTION, TEX_FILENAMES)
    set_white_diffuses()

    #----------------------------- PREPARING DATA ------------------------------
    print('Getting City Properties')
    roads, crossings, buildings, water_towers = get_city_names(CITY)
    asphalt = roads + crossings

    fix_objects(main=MAIN_OBJ, axis=CITY_AXIS)
    city_range, block_length = get_city_dims(asphalt + buildings, asphalt)
    grid = get_grid(city_range, block_length, asphalt + buildings + water_towers)

    print('Getting Skies')
    skies, sky_distribution = get_skies(SKY_DATASET_PATH, CHISQUARE_DF)

    print('Configuring Polyhaven Materials')
    mat_names = get_polyhaven_materials(city_range,
                                        enable_spec=ENABLE_SPECULARITIES,
                                        enable_trans=ENABLE_TRANSLUCENCY,
                                        enable_displacement=ENABLE_DISPLACEMENT)

    print('Adjusting Scene')
    fix_objects(ground=GROUND_PLANE, range=city_range, block=block_length)
    toggle_white_diffuses(False)

    #---------------------------- GENERATING FRAMES ----------------------------
    iterations  = NUM_FRAMES
    changes = None

    for frame in range(iterations):
        print('---------------- Frame ' + str(frame + 1) + ' / ' +\
              str(iterations) + ' -----------------')

        fix_buildings(grid, changes=changes)
        fix_objects(axis=CITY_AXIS)

        print('Updating Skies')
        sky = update_skies(SKY_DATASET_PATH, skies, sky_distribution, WORLD)

        print('Updating Ground')
        surface = update_surface(asphalt, GROUND_PLANE, CITY_AXIS, mat_names,
                                 ROAD_PROBABILITY, SIDEWALK_PROBABILITY)

        print('Updating Buildings')
        if UPDATE_BUILDINGS:
            changes = update_buildings(MAIN_OBJ, CITY_AXIS, sky, grid,
                                       city_range, block_length)

        print('Updating Main Object')
        update_object(MAIN_OBJ, OBJ_SCALE)

        print('Updating Occluders')
        update_occluders(OCCLUDERS, MAIN_OBJ, sky, OCCLUDER_AMOUNT,
                         OCCLUDER_SIZE, OCCLUDER_DISTANCE,
                         OCCLUDER_PERTURBATION)

        print('Updating Camera')
        update_camera(CAMERA, MAIN_OBJ, CAMERA_RADIUS, CAMERA_ZENITH,
                      CAMERA_PERTURBATION, sky)

        if WRITE_FRAMES and frame >= FRAME_OFFSET:
            save_keyframe(frame, OUTPUT_PATH, SCENE,
                          HALF_PRECISION_EXR, RESOLUTION, MAIN_OBJ, OCCLUDERS,
                          CITY, TERRAIN, CAMERA, WORLD, surface, sky,
                          render_full=RENDER_DEBUG_KEYFRAME)
            print('Generated frame ' + str(frame + 1) + ' / ' + str(iterations))

        print('-----------------------------------------------')


    print('DONE')

if __name__ == "__main__":
    main()
