#!/bin/sh

# Build Type
build_type=RELEASE

# CMake install prefix directory
cmake_install_dir=/usr/local

# Python3 executable path
py3_exe_path=/usr/bin/python3

# Python3 include directory
py3_include_dir=/usr/include/python3.8

# Python3 Library
# py3_lib_dir=/usr/lib/arm-linux-gnueabihf/libpython3.8m.so
py3_lib_dir=/usr/lib/python3

# Numpy include directory
np_include_dir=$HOME/.local/lib/python3.8/site-packages/numpy/core/include

if[[ ! -a ./.ENV_FLAG ]]
then
    sudo apt install python3 python3-pip
    sudo pip3 install numpy
    sudo apt install build-essential cmake git pkg-config -y
    sudo apt install libjpeg8-dev -y
    sudo apt install libtiff5-dev -y
    sudo apt install libjasper-dev -y
    sudo apt install libpng12-dev -y
    sudo apt install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev -y
    sudo apt install libgtk2.0-dev -y
    sudo apt install libatlas-base-dev gfortran -y
    touch ./.ENV_FLAG
    echo "Warning: Configure may wrong in different target platform"
    echo "Edit variables in setupCVEnv.sh"
    echo "Increase vitual memory by editing /etc/dphys-swapfile CONF_SWAPSIZE=4096"
    echo "Then rerun this script"
else
    git clone git://github.com/opencv/opencv ~/opencv
    mkdir ~/opencv/build
    cd ~/opencv/build
    cmake -D CMAKE_BUILD_TYPE=$build_type -D CMAKE_INSTALL_PREFIX=$cmake_install_dir -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -DCMAKE_SHARED_LINKER_FLAGS='-latomic' -D WITH_LIBV4L=ON PYTHON3_EXECUTABLE=$py3_exe_path PYTHON_INCLUDE_DIR=$py3_include_dir PYTHON_LIBRARY=$py3_lib_dir PYTHON3_NUMPY_INCLUDE_DIRS=$np_include_dir ..
    echo "build from $HOME/opencv/build"
    rm ./.ENV_FLAG
fi