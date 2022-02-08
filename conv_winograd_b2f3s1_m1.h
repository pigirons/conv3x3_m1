#ifndef _CONV_WINOGRAD_F3S1_M1_H
#define _CONV_WINOGRAD_F3S1_M1_H

int conv_winograd_b2f3s1_buf_size_m1(int src_h,
    int src_w,
    int channels,
    int padding_h,
    int padding_w,
    int num_outs);

int conv_winograd_b2f3s1_cvt_filter_size_m1(int channels,
    int num_outs);

void conv_winograd_b2f3s1_cvt_filter_m1(float *filter,
    int channels,
    int num_outs,
    void *buffer,
    float *cvt_filter);

void conv_winograd_b2f3s1_m1(float *src,
    int src_h,
    int src_w,
    int channels,
    int padding_h,
    int padding_w,
    float *cvt_filter,
    float *bias,
    int num_outs,
    void *buffer,
    float *dst);

#endif

