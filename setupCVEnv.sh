#!/bin/bash

# Build Type
build_type=RELEASE

# CMake install prefix directory
cmake_install_dir=/usr/local

# Python3 executable path
py3_exec_path=/usr/bin/python3

# Python3 include directory
py3_include_dir=/usr/include/python3.7

# Python3 Library
py3_lib_dir=/usr/lib/arm-linux-gnueabihf/libpython3.7m.so
#py3_lib_dir=/usr/lib/python3

# Numpy include directory
np_include_dir=$HOME/.local/lib/python3.7/site-packages/numpy/core/include

# Threads this CPU has
thread_num=4

if [ ! -e ./.ENV_FLAG ]
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
    echo "========================================================================"
    echo "Warning: Configuration may wrong in different target platform"
    echo "Configure environment variables in setupCVEnv.sh"
    echo "==> Default"
    echo "bulid_type=$build_type"
    echo "cmake_install_dir=$cmake_install_dir"
    echo "py3_exec_path=$py3_exec_path"
    echo "py3_include_dir=$py3_include_dir"
    echo "py3_lib_dir=$py3_lib_dir"
    echo "np_include_dir=$np_include_dir"
    echo "<=="
    echo "Increase vitual memory by editing /etc/dphys-swapfile CONF_SWAPSIZE=4096"
    echo "Then rerun this script"
    touch ./.ENV_FLAG
elif [[ -e ./.ENV_FLAG && ! -e ./.CONF_FLAG ]]
then
    git clone git://github.com/opencv/opencv ~/opencv
    mkdir ~/opencv/build
    cd ~/opencv/build
    cmake -D CMAKE_BUILD_TYPE=$build_type -D CMAKE_INSTALL_PREFIX=$cmake_install_dir -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -DCMAKE_SHARED_LINKER_FLAGS='-latomic' -D WITH_LIBV4L=ON PYTHON3_EXECUTABLE=$py3_exec_path PYTHON_INCLUDE_DIR=$py3_include_dir PYTHON_LIBRARY=$py3_lib_dir PYTHON3_NUMPY_INCLUDE_DIRS=$np_include_dir ..
    echo "========================================================================"
    cat /proc/cpuinfo | grep processor
    echo "Configure thread_num in setupCVEnv.sh"
    echo "==> Default"
    echo "thread_num=$thread_num"
    echo "<=="
    echo "Then rerun this script"
    touch ./.CONF_FLAG
elif [[ -e ./.ENV_FLAG && -e ./.CONF_FLAG && ! -e ./.BUILD_FLAG ]]
then
    cd ~/opencv/build
    make -j $thread_num
    sudo make install
    echo "Everything Done!"
    python3 -c "import cv2;print('OpenCV version:',cv2.__version__)"
    touch ./.BUILD_FLAG
else
    echo "OpenCV had been installed!"
    python3 -c "import cv2;print('Version:',cv2.__version__)"
fi
