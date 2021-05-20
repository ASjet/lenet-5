cmake_minimum_required(VERSION 3.5)

project(HandWrittenDightRecognition)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    /usr/local/include/eigen3
    /usr/local/include/opencv4
)

link_directories(
    /usr/local/Cellar/opencv/4.5.2_2/include/opencv4
    /usr/local/Cellar/opencv/4.5.2_2/lib
)

add_link_options(
    -lopencv_gapi
    -lopencv_stitching
    -lopencv_alphamat
    -lopencv_aruco
    -lopencv_bgsegm
    -lopencv_bioinspired
    -lopencv_ccalib
    -lopencv_dnn_objdetect
    -lopencv_dnn_superres
    -lopencv_dpm
    -lopencv_face
    -lopencv_freetype
    -lopencv_fuzzy
    -lopencv_hfs
    -lopencv_img_hash
    -lopencv_intensity_transform
    -lopencv_line_descriptor
    -lopencv_mcc
    -lopencv_quality
    -lopencv_rapid
    -lopencv_reg
    -lopencv_rgbd
    -lopencv_saliency
    -lopencv_sfm
    -lopencv_stereo
    -lopencv_structured_light
    -lopencv_phase_unwrapping
    -lopencv_superres
    -lopencv_optflow
    -lopencv_surface_matching
    -lopencv_tracking
    -lopencv_highgui
    -lopencv_datasets
    -lopencv_text
    -lopencv_plot
    -lopencv_videostab
    -lopencv_videoio
    -lopencv_viz
    -lopencv_wechat_qrcode
    -lopencv_xfeatures2d
    -lopencv_shape
    -lopencv_ml
    -lopencv_ximgproc
    -lopencv_video
    -lopencv_dnn
    -lopencv_xobjdetect
    -lopencv_objdetect
    -lopencv_calib3d
    -lopencv_imgcodecs
    -lopencv_features2d
    -lopencv_flann
    -lopencv_xphoto
    -lopencv_photo
    -lopencv_imgproc
    -lopencv_core
)

add_executable(
    test
    src/eigentest.cpp
)

add_executable(
    train
    src/train.cpp
    src/cnn.cpp
)

add_executable(
    cv2test
    src/cv2test.cpp
)


set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_COMPILER "g++-11")
add_compile_options(-O0 -g)
