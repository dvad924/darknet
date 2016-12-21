cmake_minimum_required(VERSION 2.8.12)
project(darknet)

find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  rospy
  std_msgs
  sensor_msgs
  image_transport
  camera_info_manager
  message_generation
  )
add_definitions(-DWITH_ROS -DWITH_OPENCV3 -DDEBUG)
add_message_files(
  FILES
  bbox.msg
  bbox_array.msg
  )

generate_messages(
  DEPENDENCIES
  std_msgs
  )

set(
  CUDA_NVCC_FLAGS
  ${CUDA_NVCC_FLAGS};
  -O3 -gencode arch=compute_52,code=sm_52
  )
set(CMAKE_C_FLAGS "-std=c99")

add_definitions(-DGPU -DOPENCV -DFOR_ROS -D_DEFAULT_SOURCE)

catkin_package(
  CATKIN_DEPENDS message_runtime image_transport roscpp cv_bridge sensor_msgs camera_info_manager
  )

include_directories(
  /usr/local/cuda/include
  src
  ${CUDA_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  )

link_directories(
  /usr/local/cuda/lib64
  /usr/local/lib
  )

cuda_add_executable(darknet
  src/ROS_objd.cpp
  src/gemm.c                src/utils.c
  src/cuda.c                src/convolutional_layer.c
  src/list.c                src/image.c
  src/activations.c         src/im2col.c
  src/col2im.c              src/blas.c
  src/crop_layer.c          src/dropout_layer.c
  src/maxpool_layer.c       src/softmax_layer.c
  src/data.c                src/matrix.c
  src/network.c             src/connected_layer.c
  src/cost_layer.c          src/parser.c
  src/option_list.c         src/darknet.c
  src/detection_layer.c     src/captcha.c
  src/route_layer.c         src/writing.c
  src/box.c                 src/nightmare.c
  src/normalization_layer.c src/avgpool_layer.c
  src/coco.c                src/dice.c
  src/yolo.c                src/detector.c
  src/layer.c               src/compare.c
  src/classifier.c          src/local_layer.c
  src/swag.c                src/shortcut_layer.c
  src/activation_layer.c    src/rnn_layer.c
  src/gru_layer.c           src/rnn.c
  src/rnn_vid.c             src/crnn_layer.c
  src/demo.c                src/tag.c
  src/cifar.c               src/go.c
  src/batchnorm_layer.c     src/art.c
  src/region_layer.c        src/reorg_layer.c
  src/super.c               src/voxel.c
  src/tree.c                src/deconvolutional_layer.c
  src/activation_kernels.cu src/avgpool_layer_kernels.cu
  src/blas_kernels.cu       src/col2im_kernels.cu
  src/crop_layer_kernels.cu src/convolutional_kernels.cu
  src/dropout_layer_kernels.cu src/deconvolutional_kernels.cu
  src/im2col_kernels.cu     src/maxpool_layer_kernels.cu
  src/network_kernels.cu    
  )

set_target_properties(darknet PROPERTIES COMPILE_FLAGS -pthread LINK_FLAGS -pthread)

add_dependencies(darknet darknet_generate_messages_cpp)

target_link_libraries(darknet
  m
  rt
  pthread
  stdc++
  cuda
  cudart
  cublas
  curand
  ${OpenCV_LIBRAIRES}
  ${catkin_LIBRARIES}
  )
#############
## Install ##
#############
install(TARGETS darknet
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )

