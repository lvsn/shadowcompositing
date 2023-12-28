# Compositing
This portion of our code release includes the necessary scripts to render and composite virtual objects. It assumes that the necessary assets have been unpacked in the right relative location(see the Assets section of our [main README](https://github.com/lvsn/shadowcompositing/blob/main/README.md)).

**For compositing**, we include pre-rendered EXRs ready to be used with our proposed equations. See the respective section below for more details. Alternatively, the user can render their own scenes with their own backgrounds, as detailed below. Otherwise, just skip to the compositing portion.

**For rendering**, all the camera, geometry, and illumination inputs for our test-time tutorials here are achievable via quick manual annotation (by a user), instead of requiring pre-trained estimation networks as dependencies, which are often proprietary (though those can be used instead, replacing the input preparation steps below).

## Input Preparation
Our method expects as input not just renders, but lighting, camera, and geometry estimations with respect to the background image's local scene (see Debevec's 1998 paper "Rendering Synthetic Objects into Real Scenes" for more details). This input annotation step can be done manually, with the help of tools, or with automatic inference methods (of which there are many).

- **Camera and geometry:** for simplicity and ease of use, we suggest estimating the geometry as a plane at the origin, being looked at by the camera. This can be achieved automatically by annotating parallel lines with [FSpy](https://fspy.io/) and exporting the camera to Blender with [FSpy's Blender add-on](https://github.com/stuffmatic/fSpy-Blender).

Of note, our algorithm interprets XYZ coordinates pixel-wise. Therefore, more detailed geometries can also be used instead, at the user's discretion, with possible minor adaptations to be made depending on the geometric information available.

- **Lighting estimation**: for outdoor lighting, which is our target scope, [Blender's sky nodes](https://docs.blender.org/manual/en/latest/render/shader_nodes/textures/sky.html) can be adjusted directly with sliders within Blender between steps 1 and 2 below. For an example of rendering the sky outside of Blender (which can be imported as an HDR environment map, as in the tutorial below) see our [LM Model implementation](https://github.com/lvsn/lm-model).

We further refer the user to a brief annotation and HDR compositing tutorial, dubbed [3D Object Compositing 101](https://github.com/lvsn/shadowcompositing/blob/main/src/compositing/blender/tutorial.html), adapted to this repository's needs from Prof. Jean-François Lalonde's "Algorithmic Photography" course, taught at Université Laval. Required assets are included in `demo.zip` in the `tutorial` directory. Apart from walking the user through FSpy and Blender, this tutorial can provide a solid foundation for traditional compositing, making it easier to navigate and experiment with our proposed, novel compositing pipeline.

# Rendering
1. Using the `object_insertion.blend` file included in the `blender` directory of the `demo.zip` asset file, set up a 3D scene (or use the pre-made example).
2. To render the layers correctly, open the `blender/run.py` script within Blender. Information such as objects to render and filepaths can be changed in `blender/save_annotation.py` outside of Blender, as long as `run.py` remains consistent, `save_annotation.py` is invoked by proxy.

Our Blender projects use version 3.0.1. Future version compatibility is not guaranteed.

Here's an overview of the entire process in the two above sections:
1. Set the camera in fSpy and load the fSpy file in Blender. Remember to use +Z as up in your annotation (more tips in the compositing tutorial linked above).
2. Make the desired object visible, hide others, then translate the camera and rotate/scale the object as desired. Also move the sun lamp and fine-tune the sky texture node. For best results, set the object at the world's origin.
3. Input the required fields in the `blender/save_annotation.py` script (object name in Blender, output path, scene tag). Examples are available in the code.
4. Save the script and run `run.py` from Blender.

## Proposed Compositing
- Using the provided EXR renders (or your own), adjust the filepath constants in `composite.py` (after line 275) and run the script. More details can be found in the comments within the file.
- The animation example used in `animate.py` is very similar to the previous compositing script, except it expects a whole-image detection (i.e., giving a whole-image white mask to the network as the region to make uniform). This ensures consistency as the object's shadow traverses the ground plane (more details commented in the code). The object in the code is translated in an image-based way, but a series of renders can also be used analogously.
