# CityShadows
We provide here all the necessary code to generate synthetic frames with domain randomization and guaranteed shadow interactions. A reference, asset-free Blender project can be found in the `demo.zip` file and all dataset samples (HDR rendered frames and annotations) can be found in the `datasets.zip` file. Unfortunately, we cannot provide the 3D Blender assets used to populate our scenes, as they require a [SceneCity](http://www.cgchan.com/) license.

**Note:** even without a SceneCity license, the code can be quickly adapted to other 3D contexts/scenes. Since documentation on Blender scripting can be scarce, we believe these scripts can be a valuable resource to the community as data generation needs become widespread. If this script helps your research, please refer others to our work as instructed in our [main README](https://github.com/lvsn/shadowcompositing/blob/main/README.md).

## Usage
The `generator.py` script should be run from within Blender, with a Blender Console window open to visualize the progress/logs and interrupt the generation progress if needed (otherwise, Blender just seems frozen while the script runs). Generation parameters can be set at the start of the script, as constants and flags (see the comments within the code for details).

After generation, `process_samples.py` can be used to simplify the set of Blender EXRs per scene into a single Numpy NPZ file (per sample). This is not necessary, as can be seen in our `compositing` example (where EXRs are used directly). However, since loading EXRs can take significant time, using NPZs can greatly improve training speed and is recommended when generating the data.

Other dependencies to generate the data include all publicly-available [Polyhaven textures](https://polyhaven.com/textures) (or as many as possible) as well as all [Laval HDR skies](http://hdrdb.com/sky/) bundled with parametric annotations, available publicly upon request as instructed in the dataset's website. Our Stanford Bunny OBJ is included in the `demo.zip` file and occluders are otherwise generated procedurally within Blender.
