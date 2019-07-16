#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "result_process.h"

float sigmod(float x)
{
    return 1 / (1 + exp(-x));
}

void swap_box(box& box1, box& box2)
{
    box box_buf = box1;
    box1 = box2;
    box2 = box_buf;
}

float box_iou(box& box1, box& box2)
{
    float left, right, up, down, w, h, area;
    float s1 = box1.w * box1.h;
    float s2 = box2.w * box2.h;
    float x11 = box1.x - box1.w / 2;
    float x12 = box1.x + box1.w / 2;
    float x21 = box2.x - box2.w / 2;
    float x22 = box2.x + box2.w / 2;
    float y11 = box1.y - box1.h / 2;
    float y12 = box1.y + box1.h / 2;
    float y21 = box2.y - box2.h / 2;
    float y22 = box2.y + box2.h / 2;
    left = x11 > x21 ? x11 : x21;
    right = x12 < x22 ? x12 : x22;
    up = y11 > y21 ? y11 : y21;
    down = y12 < y22 ? y12 : y22;
    w = (right - left) > 0 ? (right - left) : 0;
    h = (down - up) > 0 ? (down - up) : 0;
    area = w * h;
    return area / (s1 + s2 - area);
}




yolov3_result_parser::yolov3_result_parser(){}

void yolov3_result_parser::init(const int scale, const int width, const int height, const int num_class, const int num_anchors, const float* anchors, const int* mask, const float obj_thresh){
    this->scale = scale;
    this->num_class = num_class;
    this->num_anchors = num_anchors;
    this->width = width / scale;
    this->height = height / scale;
    this->data_len = (num_class + 4 + 1) * num_anchors * this->width * this->height;
    this->anchor_len_C = num_class + 4 + 1;
    this->obj_thresh = obj_thresh;
    this->t_obj_thresh = log(obj_thresh/(1-obj_thresh));
    this->anchors = (float*)calloc(num_anchors*2, sizeof(float));
    for(int i=0; i<num_anchors; i++)
    {
        this->anchors[i*2] = anchors[mask[i]*2] / width;
        this->anchors[i*2+1] = anchors[mask[i]*2 + 1] / height;
    }
}

yolov3_result_parser::~yolov3_result_parser()
{
    free(this->anchors);
}

void yolov3_result_parser::get_preds(float* pred_result)
{
    this->pred = pred_result;
}

float yolov3_result_parser::get_obj(int w, int h, int k_anchor)
{
    int idx = (this->anchor_len_C * k_anchor + 4) * this->width * this->height + this->width * h + w;
    float obj = this->pred[idx];
    if(obj < this->t_obj_thresh)
    {
        return 0.0;
    }else{
        obj = 1 / (1 + exp(0-obj));//sigmod
        return obj;
    }
}

void yolov3_result_parser::get_box(int w, int h, int k_anchor, box* pbox)
{
    int idx;
    float max_prob = -100000;
    int max_cls = 0;
    float cls_pred;
    //get x y w h
    idx = (this->anchor_len_C * k_anchor + 0) * this->width * this->height + this->width * h + w;
    pbox->x = (sigmod(this->pred[idx]) + w) / this->width;
    idx = (this->anchor_len_C * k_anchor + 1) * this->width * this->height + this->width * h + w;
    pbox->y = (sigmod(this->pred[idx]) + h) / this->height;
    idx = (this->anchor_len_C * k_anchor + 2) * this->width * this->height + this->width * h + w;
    pbox->w = exp(this->pred[idx]) * this->anchors[k_anchor*2];
    idx = (this->anchor_len_C * k_anchor + 3) * this->width * this->height + this->width * h + w;
    pbox->h = exp(this->pred[idx]) * this->anchors[k_anchor*2 + 1];
    for(int i=0; i<this->num_class; i++)
    {
        idx = (this->anchor_len_C * k_anchor + 5 + i) * this->width * this->height + this->width * h + w;
        cls_pred = this->pred[idx];
        if(cls_pred > max_prob)
        {
            max_cls = i;
            max_prob = cls_pred;
        }
    }
    pbox->cls = max_cls;
    pbox->prob = sigmod(max_prob);
}

void yolov3_result_parser::get_boxes()
{
    int i,j,k;
    box box_corrected;
    int cls;
    float obj;
    float prob;
    this->box_list.clear();
    for(j=0; j<height; j++)
    {
        for(i=0; i<width; i++)
        {
            for(k=0; k<this->num_anchors; k++)
            {
                obj = this->get_obj(i, j, k);
                if(obj > this->obj_thresh)
                {
                    this->get_box(i, j, k, &box_corrected);
                    this->box_list.push_back(box_corrected);
                }
            }
        }
    }
}

