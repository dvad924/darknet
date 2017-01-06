#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include <sys/time.h>

#define FRAMES 3

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
     #ifdef WITH_OPENCV3
     #include "opencv2/videoio/videoio_c.h"
     #endif

#ifdef WITH_ROS
IplImage * get_Ipl_image(void);
#endif

image ipl_to_image(IplImage*src);

image get_image_from_stream(CvCapture *cap);

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static float **probs;
static box *boxes;
static network net;
static image in   ;
static image in_s ;
static image det  ;
static image det_s;
static image disp = {0};
static CvCapture * cap;
static float fps = 0;
static float demo_thresh = 0;

static float *predictions[FRAMES];
static int demo_index = 0;
static image images[FRAMES];
static float *avg;
static int delay;
static int frame_skip;


void *fetch_in_thread(void *ptr)
{
    in = get_image_from_stream(cap);
    if(!in.data){
        error("Stream closed.");
    }
    in_s = resize_image(in, net.w, net.h);
    return 0;
}
#ifdef WITH_ROS
/* This method will be used to capture
   input from a ros node, convert it to
   the struct image format for use in 
   the detector*/
static ROS_box * ROI_boxes;

void *fetch_in_thread_ros(void* ptr)
{

  IplImage* src = get_Ipl_image();
  image im = ipl_to_image(src);
  free(src);
  rgbgr_image(im);
  in = im;
  in_s = resize_image(in,net.w,net.h);
  return 0;
}

void *detect_in_thread_ros(void *ptr)
{
    float nms = .4;

    layer l = net.layers[net.n-1];
    
    float *X = in_s.data;
    
    float *prediction = network_predict(net, X);

    //memcpy(predictions[demo_index], prediction, l.outputs*sizeof(float));

    //mean_arrays(predictions, FRAMES, l.outputs, avg);

    //l.output = avg;

    free_image(in_s);
    if(l.type == DETECTION){
      fprintf(stderr,"Getting Detection Boxes");
        get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
    } else if (l.type == REGION){
        get_region_boxes(l, 1, 1, demo_thresh, probs, boxes, 0, 0);
    } else {
        error("Last layer must produce detections\n");
    }
    if (nms > 0) do_nms(boxes, probs, l.w*l.h*l.n, l.classes, nms);
    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n",fps);
    printf("Objects:\n\n");

    
    //extract bounding boxes put in array
    int total = l.h * l.w *l.n;
    int i,j;
    int count = 0;
    for(i = 0; i < total; ++i)
      {
        //iterate through possible boxes and collect the bounding boxes
        for(j = 0; j < l.classes; ++j)
          {
            float x_center = 0;
            float y_center = 0;
            float bbox_width = 0;
            float bbox_height= 0;
  
            if(probs[i][j])
              {
                box b = boxes[i];
                float xmin = b.x - b.w/2.0;
                float xmax = b.x + b.w/2.0;
                float ymin = b.y - b.h/2.0;
                float ymax = b.y + b.h/2.0;

                if (xmin < 0) xmin = 0;
                if (ymin < 0) ymin = 0;
                if (xmax > 1) xmax = 1;
                if (ymax > 1) ymax = 1;

                x_center = (xmin+xmax)/2;
                y_center = (ymin+ymax)/2;
                bbox_width = xmax - xmin;
                bbox_height= ymax - ymin;            
              

            // define bbox/

                ROI_boxes[count].x = x_center;
                ROI_boxes[count].y = y_center;
                ROI_boxes[count].w = bbox_width;
                ROI_boxes[count].h = bbox_height;
                ROI_boxes[count].Class = j;
                count++; 
              }
          }
      }
    //create array to store found boxes
    // if no object detected, make sure that num = 0
    if (count == 0)
      {
        ROI_boxes[0].num = 0;
      }
    else
      {
        ROI_boxes[0].num = count;
      }
    printf("Num Objs: %d\n",count);
}
#endif
void *detect_in_thread(void *ptr)
{
    float nms = .4;

    layer l = net.layers[net.n-1];
    float *X = det_s.data;
    fprintf(stderr,"Buffer Size : %d\n", sizeof(X)/sizeof(X[0]));
    float *prediction = network_predict(net, X);

    memcpy(predictions[demo_index], prediction, l.outputs*sizeof(float));
    mean_arrays(predictions, FRAMES, l.outputs, avg);
    l.output = avg;

    free_image(det_s);
    if(l.type == DETECTION){
        get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
    } else if (l.type == REGION){
        get_region_boxes(l, 1, 1, demo_thresh, probs, boxes, 0, 0);
    } else {
        error("Last layer must produce detections\n");
    }
    if (nms > 0) do_nms(boxes, probs, l.w*l.h*l.n, l.classes, nms);
    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.1f\n",fps);
    printf("Objects:\n\n");

    images[demo_index] = det;
    det = images[(demo_index + FRAMES/2 + 1)%FRAMES];
    demo_index = (demo_index + 1)%FRAMES;

    draw_detections(det, l.w*l.h*l.n, demo_thresh, boxes, probs, demo_names, demo_alphabet, demo_classes);

    return 0;
}

double get_wall_time()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

#ifdef WITH_ROS
void ros_load_network(char *cfgfile, char* weightfile, float thresh, int cam_index, const char * filename, char **names, int classes, int f_skip)
{
  image **alphabet = 0;


  demo_names = names;
  demo_alphabet = alphabet;
  demo_classes = classes;
  demo_thresh = thresh;
  printf("Demo\n");
  net = parse_network_cfg(cfgfile);
  if(weightfile){
    load_weights(&net, weightfile);
  }
  set_batch_network(&net, 1);
  srand(2222222);
  
  layer l = net.layers[net.n-1];
  int j;

  avg = (float *) calloc(l.outputs, sizeof(float));
  for(j = 0; j < FRAMES; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));
  for(j = 0; j < FRAMES; ++j) images[j] = make_image(1,1,3);

  boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
  ROI_boxes = (ROS_box *)calloc(l.w*l.h*l.n,sizeof(ROS_box));
  probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
  for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));

  
}

ROS_box* ros_demo()
{

  double before = get_wall_time();
  fetch_in_thread_ros(0);
  
  detect_in_thread_ros(0);
  
  disp = in;
  det = in;
  free_image(disp);


  
  double after = get_wall_time();
  float curr = 1./(after - before);
  fps = curr;
  before = after;
  return ROI_boxes;

}
#endif

void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix)
{
    //skip = frame_skip;
    image **alphabet = load_alphabet();
    int delay = frame_skip;
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    printf("Demo\n");
    net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);

    srand(2222222);

    if(filename){
        printf("video file: %s\n", filename);
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);
    }

    if(!cap) error("Couldn't connect to webcam.\n");

    layer l = net.layers[net.n-1];
    int j;

    avg = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < FRAMES; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < FRAMES; ++j) images[j] = make_image(1,1,3);

    boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
    probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));

    pthread_t fetch_thread;
    pthread_t detect_thread;

    fetch_in_thread(0);
    det = in;
    det_s = in_s;

    fetch_in_thread(0);
    detect_in_thread(0);
    disp = det;
    det = in;
    det_s = in_s;

    for(j = 0; j < FRAMES/2; ++j){
        fetch_in_thread(0);
        detect_in_thread(0);
        disp = det;
        det = in;
        det_s = in_s;
    }

    int count = 0;
    if(!prefix){
        cvNamedWindow("Demo", CV_WINDOW_NORMAL); 
        cvMoveWindow("Demo", 0, 0);
        cvResizeWindow("Demo", 1352, 1013);
    }

    double before = get_wall_time();

    while(1){
        ++count;
        if(1){
            if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
            if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");

            if(!prefix){
                show_image(disp, "Demo");
                int c = cvWaitKey(1);
                if (c == 10){
                    if(frame_skip == 0) frame_skip = 60;
                    else if(frame_skip == 4) frame_skip = 0;
                    else if(frame_skip == 60) frame_skip = 4;   
                    else frame_skip = 0;
                }
            }else{
                char buff[256];
                sprintf(buff, "%s_%08d", prefix, count);
                save_image(disp, buff);
            }

            pthread_join(fetch_thread, 0);
            pthread_join(detect_thread, 0);

            if(delay == 0){
                free_image(disp);
                disp  = det;
            }
            det   = in;
            det_s = in_s;
        }else {
            fetch_in_thread(0);
            det   = in;
            det_s = in_s;
            detect_in_thread(0);
            if(delay == 0) {
                free_image(disp);
                disp = det;
            }
            show_image(disp, "Demo");
            cvWaitKey(1);
        }
        --delay;
        if(delay < 0){
            delay = frame_skip;

            double after = get_wall_time();
            float curr = 1./(after - before);
            fps = curr;
            before = after;
        }
    }
}
#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

