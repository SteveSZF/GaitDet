#ifndef RESULT_PROCESS_
#define RESULT_PROCESS_

#include <iostream>
#include <vector>

using namespace std;

typedef struct box{
    int cls;
    float x;
    float y;
    float w;
    float h;
    float prob;
}box;

float box_iou(box& box1, box& box2);
void swap_box(box& box1, box& box2);

float sigmod(float x);



class yolov3_result_parser
{
public:
    yolov3_result_parser();
    //yolov3_result_parser(int scale, int width, int height, int num_class, int num_anchors, float* anchors, int* mask, float obj_thresh);
    ~yolov3_result_parser();
    void get_preds(float* pred_result);
    float get_obj(int w, int h, int k_anchor);
    void get_box(int w, int h, int k_anchor, box* pbox);
    void get_boxes();
    void init(const int scale, const int width, const int height, const int num_class, const int num_anchors, const float* anchors, const int* mask, const float obj_thresh);


    int num_class;
    int num_anchors;
    int width;
    int height;
    int scale;
    int data_len;
    int anchor_len_C;
    float obj_thresh;
    float t_obj_thresh;
    float* anchors;
    float* pred;
    vector<box> box_list;
};


#endif
