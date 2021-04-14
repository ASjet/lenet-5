#!/bin/sh
sudo apt install build-essential cmake git pkg-config -y 
sudo apt install libjpeg8-dev -y
sudo apt install libtiff5-dev -y
sudo apt install libjasper-dev -y
sudo apt install libpng12-dev -y
sudo apt install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev -y
sudo apt install libgtk2.0-dev -y
sudo apt install libatlas-base-dev gfortran -y
git clone git://github.com/opencv/opencv ~/opencv
mkdir ~/opencv/build
cd ~/opencv/build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D CMAKE_SHARED_LINKER_FLAGS='-latomic' -D WITH_LIBV4L=ON ..
echo "build from $HOME/opencv/build"
