# DIWU

This C++ code (and MATLAB wrapper) provides a minimal working example (MWE) of the paper "Efficient Sliding Window Computation for
NN-Based Template Matching", ECCV 2018, by Lior Talker, Yael Moses and Ilan Shimshoni.

For questions or information about the paper or code please 
contact Lior Talker (liortalker@gmail.com or http://liortalker.wix.com/liortalker)
Only for academic or other non-commercial purposes (under GPL terms).

Running the algorithm:

Invoke runExample.m in MATLAB. 
Due to lack of space, deep features of the VGG network are not included here. 
To download the network to be used with our code please see the project page: https://github.com/roimehrez/DDIS,
for the paper "Template matching with deformable diversity similarity" by I. Talmi, R. Mechrez, and L. Zelnik-Manor.
We thank the authors for their public code which helped us implement our method.

Dependencies:

The provided MATLAB wrapper includes a mex for Windows 64 bit.
To build the C++ sources for other platforms, use the sources in \DIWU_C++_src.
These sources are dependent on opencv 2.4 which is also included in the project as the compressed file mex_opencv_2.4.rar.
