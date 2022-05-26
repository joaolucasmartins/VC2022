# VC2022

A simple traffic sign detection and classification project that uses classic Image Processing methods. Deveped in the context of the C.U. V.C.

## Code

The frs.py is an fast radial symetry implementation in python, found [here](https://github.com/ChristianGutowski/frst_python), which
was adapted by the group to serve our needs.
The notebook contains the entire identification process.

## Images

The used dataset can be found in kaggle [Here](https://www.kaggle.com/datasets/andrewmvd/road-sign-detection).
For the notebook to run, the dataset must be extracted to the dataset/ directory.

Any extra images need to be placed within the dataset/images/ directory, and the notebook function evaluateImage must
be called with the desired image name
