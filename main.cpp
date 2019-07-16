#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>
#include <pthread.h>
#include <sched.h>
#include<semaphore.h>
#include<opencv2/tracking.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/video.hpp>
#include<opencv2/core/utility.hpp>

#include "deeplab.cuh"
#include "Yolov3.h"
using namespace std;
using namespace cv;
const float anchors[18] = { 12,15,  18,32,  35,25,  32,63,  64,47,  61,121,  118,92,  158,200,  375,328};
//const float anchors[18] = { 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326};
const int mask[3][3] = {
    {6,7,8},
    {3,4,5},
    {0,1,2}
};
static void *semantic_segmentation (void *arg);
static void *person_detection (void *arg);
static void *person_tracking (void *arg);
static sem_t sem_deeplab;
static sem_t sem_yolov3;
static sem_t sem_track;
static sem_t sem_main;
static cv::Mat img;
static vector<box> box_list, box_result;
static bool running = true;
const char *track_alg = "KCF";
const int resized_width = 200;

int main(int argc, char** argv)
{
    if(argc < 3){
        std::cerr << "more args needed" << std::endl;
        return -1; 
    }
    
    cv::VideoCapture reader(argv[1]);
    
    reader.read(img);
    const int width = img.cols;
    const int height = img.rows; 
    DeepLab deeplab(width, height, argv[2]);

    const float obj_thresh = 0.9, nms_thresh = 0.1; 
    Yolov3 yolo(width, height, anchors, mask, obj_thresh, nms_thresh, argv[3]);
    vector<box> tmp;
    yolo.doInference(img, box_list, tmp);
    box_result = tmp;
    for(int i = 0; i < 5; i++){
        reader.read(img);
        yolo.doInference(img, box_list, tmp);
        if(box_result.size() < tmp.size()) box_result = tmp;
    }
    
    vector<Rect2d> objs;
    vector<Ptr<Tracker>> algorithms;
    std::cout << box_result.size() << std::endl;
    for(int i = 0; i < box_result.size(); i++){
        int px = (box_result[i].x - box_result[i].w / 2) * width;
        int py = (box_result[i].y - box_result[i].h / 2) * height;
        int pw = box_result[i].w * width;
        int ph = box_result[i].h * height;
       
        objs.push_back(Rect(px, py, pw, ph));
        algorithms.push_back(TrackerMOSSE::create());
        //algorithms.push_back(TrackerKCF::create());
        //algorithms.push_back(TrackerMIL::create());
    }
    //MultiTracker trackers(track_alg);
    MultiTracker trackers;
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(resized_width, height * resized_width / width));
    trackers.add(algorithms, resized_img, objs); 
        
    pthread_t seg_t;
    pthread_t det_t;
    pthread_t trk_t;
    
    sem_init (&sem_deeplab, 0, 0);
    sem_init (&sem_yolov3, 0, 0);
    sem_init (&sem_track, 0, 0);
    sem_init (&sem_main, 0, 0);
    
    int ret1 = pthread_create(&seg_t, NULL, semantic_segmentation, (void *)&deeplab );
    if (ret1 != 0){
        printf("fail to create threads\n");
        return 1;
    }
    int ret2 = pthread_create(&det_t, NULL, person_detection, (void *)&yolo);
    if (ret2 != 0){
        printf("fail to create threads\n");
        return 1;
    }
    int ret3 = pthread_create(&trk_t, NULL, person_tracking, (void *)&trackers);
    if (ret3 != 0){
        printf("fail to create threads\n");
        return 1;
    }


    cv::namedWindow("Gait", cv::WINDOW_NORMAL);
    cv::resizeWindow("Gait", deeplab.get_OUTPUT_W(), deeplab.get_OUTPUT_H());

    cv::TickMeter tm, tp; 
    while(running){
        tm.reset(); tm.start();
	if(!reader.read(img)){
	    sem_post(&sem_deeplab);
            sem_post(&sem_yolov3);
            sem_post(&sem_track);
            running = false;
            break;
        }
        

        //yolo.doInference(img, box_list, box_result);
	//deeplab.doInference(img);
        sem_post(&sem_deeplab);
        sem_post(&sem_yolov3);
        sem_post(&sem_track);

        sem_wait(&sem_main);
        sem_wait(&sem_main);
        sem_wait(&sem_main);

        vector<box> track_boxes;
        
        for (int j = 0; j < trackers.getObjects().size(); j++) {
            //box trk_box;
            //trk_box.x = trackers.getObjects()[j].tl()[] + trackers.getObjects()[j].tl()
            Rect ob = trackers.getObjects()[j];
            float max_iou = 0.0;
            int idx = -1;
            for(int k = 0; k < box_result.size(); k++){
                double px = (box_result[k].x - box_result[k].w / 2) * width;
                double py = (box_result[k].y - box_result[k].h / 2) * height;
                double pw = box_result[k].w * width;
                double ph = box_result[k].h * height;
                
                Rect tmp(px, py, pw, ph);
                Rect intersect_area = tmp & ob;   
                Rect union_area = tmp | ob;
                if((float)intersect_area.area() / union_area.area() >= max_iou){
                    max_iou = (float)intersect_area.area() / union_area.area();
                    idx = k;
                }
            }
            track_boxes.push_back(box_result[idx]);
            box_result.erase(box_result.begin() + idx);
            //cout << trackers.getObjects()[j].tl() << endl;
        }
        yolo.make_output(deeplab.real_out, track_boxes);
	    
	cv::imshow("Gait", deeplab.real_out);
        cv::waitKey(2);
        tm.stop();
        std::cout << "time for per frame: " << tm.getTimeMilli() << std::endl;
    }
    sem_destroy(&sem_deeplab);
    sem_destroy(&sem_yolov3);
    sem_destroy(&sem_track);
    sem_destroy(&sem_main);
    destroyWindow("Gait");
    reader.release();
    return 0;
}

void *semantic_segmentation (void *arg)
{
    DeepLab *deeplab = (DeepLab *)arg;
    while(running)
    {

        sem_wait(&sem_deeplab);

        deeplab->doInference(img);
        //printf("a=%d, ", a);

        sem_post(&sem_main);
    }
}
void *person_detection(void *arg)
{
    Yolov3 *yolo = (Yolov3*)arg;
    while(running)
    {
        sem_wait(&sem_yolov3);
            
        yolo->doInference(img, box_list, box_result);
        //printf("b=%d\n", b);

        sem_post(&sem_main);
    }
}
void *person_tracking(void *arg)
{
    MultiTracker *tracker = (MultiTracker*)arg;
    cv::Mat resized_img;
    const int width = img.cols;
    const int height = img.rows;
   
    cv::TickMeter tm;
    while(running)
    {
        sem_wait(&sem_track);
        cv::resize(img, resized_img, cv::Size(resized_width, height * resized_width / width));
        tm.reset(); tm.start();
        tracker->update(resized_img);
        //printf("b=%d\n", b);
        tm.stop();
        cout << "tracking time: " << tm.getTimeMilli() << endl;
        sem_post(&sem_main);
    }
}

