# test sgm on windows

# Requirements
libSGM needs CUDA (compute capabilities >= 3.0) to be installed.  
Moreover, to build the sample, we need the following libraries:
- OpenCV 3.0 or later
- CMake 3.1 or later
- vs2017 or vs2019
- MINGW64

# RUN


- mkdir build
- cd build
- cmake .. -G"Visual Studio 15 2017 Win64"; or cmake .. -G"Visual Studio 16 2019 Win64"
 
show disparity by matlab:

<img src="./image/1.png" alt="disparity" width="45%">

show point cloud by meshlab:
<img src="./image/2.png" alt=" point cloud" width="45%">


