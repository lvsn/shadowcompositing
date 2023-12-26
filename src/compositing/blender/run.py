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

# Note: open and run just this script from inside Blender. This way,
#       save_annotation.py can be changed outside Blender's editor.

DIRECTORY = 'your_path_here' # Path to the save_annotation.py script
SCRIPT = 'save_annotation.py'

FILEPATH = os.path.join(DIRECTORY, SCRIPT)
NAMESPACE = {"__file__": FILEPATH, "__name__": "__main__"}

with open(FILEPATH, 'rb') as file:
    exec(compile(file.read(), FILEPATH, 'exec'), NAMESPACE)
