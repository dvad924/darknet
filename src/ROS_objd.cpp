#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/Point.h>
#include <vector>
#include <string>
#include <iostream>
#include <pthread.h>
#include <std_msgs/Int8.h>
#include <math.h>
#include <darknet/bbox_array.h>
#include <darknet/bbox.h>
#include "ROS_interface.h"

extern "C" {
  #include "box.h"
  #include "demo.h"
  #include "data.h"
  #include "option_list.h"
}



// define yolo inputs



const std::string class_labels[] = { "Face" };


const int num_classes = sizeof(class_labels)/sizeof(class_labels[0]);

cv::Mat cam_image_copy;

// define parameters
const std::string CAMERA_TOPIC_NAME = "/camera/image_raw";
const std::string CAMERA_TOPIC_INFO = "/camera/camera_info";
const std::string OPENCV_WINDOW = "WINDOW";
int FRAME_W = 640;
int FRAME_H = 480;
int FRAME_AREA;
int FRAME_COUNT = 0;

// define a function that will replace cvVideoCapture.
// This function is called in yolo_kernels and allows YOLO to receive the ROS image
// message as an IplImage
extern "C" IplImage* get_Ipl_image()
{
  IplImage* ROS_img = new IplImage(cam_image_copy);
  return ROS_img;
}

inline std::string badDefaultCheck(std::string in, std::string bad, std::string df)
{
  if ( in == bad )
    {
      return df;
    }
  else
    return in;
}

class ObjectDetector
{
  //We want a subscriber that will listen and
  //consume resources from a camera
  //for speed and efficiency we will use compressed
  //images
  ros::NodeHandle                     _nh;
  image_transport::ImageTransport     _it;
  image_transport::Subscriber         _sub;
  image_transport::Publisher          _im_pub;
  ros::Publisher                      _pub;
  ros::Publisher                      _bb_pub;
  std::vector<std::vector<ROS_box> >  _class_bboxes;
  std::vector<int>                    _class_obj_count;
  std::vector<cv::Scalar>             _bbox_colors;
  darknet::bbox_array                 _bbox_results_msg;
  ROS_box*                            _boxes;
  
public:
  ObjectDetector() : _it(_nh), _class_bboxes(num_classes),
                     _class_obj_count(num_classes,0),
                     _bbox_colors(num_classes)
  {

    int incr = floor(255/num_classes);
    for(int i = 0; i < num_classes; i++)
      {
        _bbox_colors[i] = cv::Scalar(255 - incr* i, 0 + incr*i,
                                     255 - incr*i);
      }

    std::string incam    = _nh.resolveName( "incam" );
    std::string objcount = _nh.resolveName( "objcout" );
    std::string bbox     = _nh.resolveName( "bbox" );
    std::string outcam   = _nh.resolveName( "outcam" );
    incam    = badDefaultCheck( incam, "/incam", CAMERA_TOPIC_NAME );
    objcount = badDefaultCheck( objcount, "/objcount", "/found_objects" );
    bbox     = badDefaultCheck( bbox , "/bbox" , "/darknet_bboxes" );
    outcam   = badDefaultCheck( outcam, "/outcam", "/darknet_cam" );
      
    _sub = _it.subscribe( incam , 1 ,
                               &ObjectDetector::cameraCallback , this );

    _pub    = _nh.advertise<std_msgs::Int8>( objcount , 1 );
    _bb_pub = _nh.advertise<darknet::bbox_array>( bbox , 1 );
    _im_pub = _it.advertise( outcam , 1 );

  }
  ~ObjectDetector()
  {

  }

private:
  void parseBBoxes(cv::Mat &input_frame, std::vector<ROS_box> &class_boxes,
                   int &class_obj_count, cv::Scalar &bbox_color,
                   const std::string &class_label)
  {
    darknet::bbox bbox_result;
    ROS_INFO("BOXES: %d\n",class_obj_count);
    for (int i = 0; i < class_obj_count; i++)
      {
        int xmin = (class_boxes[i].x - class_boxes[i].w/2)*FRAME_W;
        int ymin = (class_boxes[i].y - class_boxes[i].h/2)*FRAME_H;
        int xmax = (class_boxes[i].x + class_boxes[i].w/2)*FRAME_W;
        int ymax = (class_boxes[i].y + class_boxes[i].h/2)*FRAME_H;

        bbox_result.Class = class_label;
        bbox_result.xmin = xmin;
        bbox_result.ymin = ymin;
        bbox_result.xmax = xmax;
        bbox_result.ymax = ymax;
        _bbox_results_msg.bboxes.push_back(bbox_result);

        #ifdef DEBUG
        cv::Point topLeftCorner = cv::Point(xmin,ymin);
        cv::Point botRightCorner = cv::Point(xmax,ymax);
        cv::rectangle(input_frame,topLeftCorner, botRightCorner, bbox_color,2);
        cv::putText(input_frame,class_label, cv::Point(xmin, ymax+15),
                    cv::FONT_HERSHEY_PLAIN, 1.0, bbox_color, 2.0);
                    
        #endif
      }
  }
  
  void run(cv::Mat& img){
    cv::Mat input_frame = img.clone();
    
    _boxes = ros_demo();

    //get number of boxes
    int num = _boxes[0].num;
    ROS_INFO("Num Detections : %d\n",num);
    //if at least one bbox found draw_boxes
    if (num > 0 && num <= 100)
      {
        std::cout << "#Objects: " << num << std::endl;

        //split bounding boxes by class
        for( int i = 0; i< num; i++)
          {
            for( int j = 0; j < num_classes; j++)
              {
                if( _boxes[i].Class == j)
                  {
                    _class_bboxes[j].push_back(_boxes[i]);
                    _class_obj_count[j]++;
                  }
              }
          }
        //send message that object has been detected
        std_msgs::Int8 msg;
        msg.data = 1;
        _pub.publish(msg);

        for(int i = 0; i< num_classes; i++)
          {
            if(_class_obj_count[i] > 0) parseBBoxes(input_frame,_class_bboxes[i],
                                                    _class_obj_count[i],
                                                    _bbox_colors[i],
                                                    class_labels[i]);
          }

        ros::Time ct = ros::Time::now();
        _bbox_results_msg.header.stamp = ct;
        _bb_pub.publish(_bbox_results_msg);
        std_msgs::Header header = std_msgs::Header();
        header.stamp = ct;
        sensor_msgs::ImagePtr im_msg = cv_bridge::CvImage(header,"bgr8", cam_image_copy).toImageMsg();
        _im_pub.publish(im_msg);
        _bbox_results_msg.bboxes.clear();
      }
    else
      {
        ROS_INFO("PUBLISHING ALL\n");
        std_msgs::Int8 msg;
        msg.data = 0;
        _pub.publish(msg);
        ros::Time ct = ros::Time::now();
        _bbox_results_msg.header.stamp = ct;
        _bb_pub.publish(_bbox_results_msg);
        std_msgs::Header header = std_msgs::Header();
        header.stamp = ct;
        sensor_msgs::ImagePtr im_msg = cv_bridge::CvImage(header,"bgr8", cam_image_copy).toImageMsg();
        _im_pub.publish(im_msg);
        _bbox_results_msg.bboxes.clear();
        
      }

    for (int i = 0; i< num_classes; i++)
      {
        _class_bboxes[i].clear();
        _class_obj_count[i] = 0;
      }
    //Free the memory used
    input_frame.release();
  }
  void infoCallback(const sensor_msgs::CameraInfoPtr& info)
  {
    std::cout << "Info Received" << std::endl;
    std::cout << info->height << std::endl;
    std::cout << info->width << std::endl;
  }
  
  void cameraCallback(const sensor_msgs::ImageConstPtr& msg)
  {
    std::cout << "USB Image Received" << std::endl;
    cv_bridge::CvImagePtr cam_image;

    try
      {
        cam_image = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::BGR8);
        
        
      }
    catch (cv_bridge::Exception& e)
      {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
      }
    if (cam_image)
      {
        cam_image_copy = cam_image->image.clone();
        FRAME_W = cam_image->image.cols;
        FRAME_H = cam_image->image.rows;
        run(cam_image->image);
      }
    else
      {
        fprintf(stderr, "No Image\n");
      }
    
  }
  int dummy()
  {
    return 1;
  }
};

#ifdef WITH_ROS
int main(int argc, char** argv)
{
  ROS_INFO("I LIVE\n");
  ros::init(argc, argv, "listener");

  char *cfg = "/home/avail/code/darknet/cfg/face.cfg";
  char *datacfg = "/home/avail/code/darknet/cfg/face.data";
  list *options = read_data_cfg(datacfg);
  char *weights = "/home/avail/backup/face_20000.weights";
  float thresh = 0.25;
  int index = 0;
  const char*filename = 0;
  int classes = option_find_int(options, "classes",20);
  ROS_INFO("NUM_CLASSES: %d\n",classes);
  char * name_list = option_find_str(options,"names","data/names.list");
  ROS_INFO("Names : %s\n",name_list);
  char **names = get_labels(name_list);
  int f_skip = 1;
  ROS_INFO("LOADING... \n%s\n%s\n%s\n",cfg,datacfg,weights);
  //load the network into memory based on the input files
  ros_load_network(cfg,weights,thresh,index,filename, names, classes,f_skip);


  
  ObjectDetector od;
  ros::spin();
  return 0;
}
#endif
