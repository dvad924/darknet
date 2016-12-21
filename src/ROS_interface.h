#ifndef ROS_INTERFACE
#define ROS_INTERFACE
#include <opencv2/core/core_c.h>
#ifdef __cplusplus
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#endif


typedef struct {
  float x, y, w, h;
  int num, Class;
} ROS_box;

#endif
