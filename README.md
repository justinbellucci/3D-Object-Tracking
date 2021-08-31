# 3D Object Tracking

This final project for Udacity's Sensor Fusion Camera course demonstrates various methods to track a 3D object using keypoint detection and feature matching, lidar point cloud data, and camera imagery for classification using the YOLO deep learning model. Estimation of the time to collision is calulated using the constant velocity model.

<img src="images/Yolo_traffic.gif" width="800"/>
<img src="images/Kypt_traffic.gif" width="800">

## Dependencies for Running Locally

The following dependencies are required to run the program locally.
* cmake >= 3.17
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.3 (Linux, Mac)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
* OpenCV >= 4.5
  * The OpenCV 4.5 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* NOTE: This project is tested using Mac OSX 10.15

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.