# Shadow Harmonization for Realistic Compositing
This repository contains the official implementation of the SIGGRAPH Asia 2023 proceeding "Shadow Harmonization for Realistic Compositing." Code is available to train and test the network. Example scripts for object compositing, animation, rendering, and dataset generation are also included. Links for assets such as proposed datasets, pre-trained models, and Blender project files are listed below.

- **License:** Creative Commons Attribution Non-Commercial 4.0 International  (CC BY-NC 4.0)
- **For technical questions and feedback**, feel free to contact Lucas Valença (*lucas@valenca.io*).
- **For other inquiries**, please contact Prof. Jean-François Lalonde (*jflalonde@gel.ulaval.ca*).

The manuscript, supplemental material, slides, and fast-forward video are all publicly available at the project website (*https://lvsn.github.io/shadowcompositing*). [Open access at the ACM Digital Library](https://dl.acm.org/doi/10.1145/3610548.3618227) is also available.

This work was developed at Université Laval in collaboration with Adobe.

If you use this code, feel free to cite it with the following BibTeX:
```
  @inproceedings{valenca2023shadow,
                 title     = {Shadow Harmonization for Realistic Compositing},
                 author    = {Valen{\c{c}}a, Lucas and Zhang, Jinsong and Gharbi, Micha{\"e}l and Hold-Geoffroy, Yannick and Lalonde, Jean-Fran{\c{c}}ois},
                 booktitle = {ACM SIGGRAPH Asia 2023 Conference Proceedings},
                 year      = {2023}
  }
```

## Code Structure
The source code of this project can be found in the `src` directory. Within it, there are 4 separate sub-projects, as listed below. Each sub-project directory contains a separate README with the necessary specifics.

 - `docker`: contains the necessary scripts to create and run the Docker image used to train the network. This simplifies compatibility issues when training and testing our PyTorch code. For usage information, see our [Docker README](https://github.com/lvsn/shadowcompositing/blob/main/src/docker/README.md).
 - `network`: contains the main source code for the deep generative pipeline. For more information, see the [Network README](https://github.com/lvsn/shadowcompositing/blob/main/src/network/README.md).
- `compositing`: contains examples of compositing (including the teaser image) as well as an animation example. There is also an example of how to render a scene that is suitable for our compositing code using Blender at `compositing/blender`, which is introduced with a tutorial on traditional compositing, *3D Object Compositing 101*. See the [Compositing README](https://github.com/lvsn/shadowcompositing/blob/main/src/compositing/README.md) for more information.
- `city_shadows`: contains the generation and post-processing code to re-create our (publicly available) synthetic dataset. See the [CityShadows README](https://github.com/lvsn/shadowcompositing/blob/main/src/city_shadows/README.md) for more information.

For further questions, feel free to reach out using the previously-listed emails.

## Assets
The resources for this project should be placed in the `rsc` directory, as the scripts in `src` are configured to work with the provided directory structure. Within `rsc`, there are 3 directories:

 - `checkpoints`: directory for the pre-trained weights for our proposed models. Simply unzip `checkpoints.zip` into it. [The ZIP file can be downloaded here.](https://hdrdb-public.s3.valeria.science/shadowcompositing/checkpoints.zip)
 - `datasets`: directory for our training data, real and synthetic, including our proposed augmentations and ground truth annotations. Simply unzip `datasets.zip` into it. [The ZIP file can be downloaded here.](https://hdrdb-public.s3.valeria.science/shadowcompositing/datasets.zip)
 - `demo`: rendered layers for two separate scenes to be used for our compositing and animation examples. Also includes a sample Blender project with the necessary rendering setup and scripts, and assets for our tutorial *3D Object Compositing 101* (see the [Compositing README](https://github.com/lvsn/shadowcompositing/blob/main/src/compositing/README.md)). Simply unzip `demo.zip` into it. [The ZIP file can be downloaded here.](https://hdrdb-public.s3.valeria.science/shadowcompositing/demo.zip)
