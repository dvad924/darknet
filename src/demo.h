#ifndef DEMO
#define DEMO

#include "image.h"
#include "ROS_interface.h"
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix);

void ros_load_network(char *cfgfile, char* weightfile, float thresh, int cam_index, const char * filename, char **names, int classes, int f_skip);

ROS_box * ros_demo();

#endif
